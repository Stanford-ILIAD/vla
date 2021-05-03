/**
 * velocityControl.cpp
 *
 * A Minimal Script for Controlling the Robot using Joint Velocity Commands; the Robot sends back Position, Velocity,
 * Torque, and the Jacobian.
 */

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <sstream>
#include <franka/exception.h>
#include <franka/robot.h>
#include <franka/model.h>

class VelocityController {
  public:
    VelocityController(franka::Model& model, int port);
    franka::JointVelocities operator()(const franka::RobotState& robot_state, franka::Duration period);
  private:
    franka::Model* modelPtr;
    double time = 0.0;
    std::array<double, 100> buffer1;
    std::array<double, 100> buffer2;
    std::array<double, 100> buffer3;
    std::array<double, 100> buffer4;
    std::array<double, 100> buffer5;
    std::array<double, 100> buffer6;
    std::array<double, 100> buffer7;
    std::array<double, 100> bufferRate;
    std::array<double, 7> qdot;
    std::array<double, 7> control_input;
    double MovingAverage(std::array<double, 100>& buffer, double input);
    int sock = 0;
    int valread;
    struct sockaddr_in serv_addr;
    char buffer[200] = {0};
    int steps = 0;
};
VelocityController::VelocityController(franka::Model& model, int port) {
  modelPtr = &model;
  for (int i = 0; i < 7; i++) {
    control_input[i] = 0.0;
  }
  for (int i = 0; i < 100; i++) {
    buffer1[i] = 0.0;
    buffer2[i] = 0.0;
    buffer3[i] = 0.0;
    buffer4[i] = 0.0;
    buffer5[i] = 0.0;
    buffer6[i] = 0.0;
    buffer7[i] = 0.0;
    bufferRate[i] = 0.0;
  }
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    std::cout << "Socket Creation Error!" << std::endl;
  }
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(port);
  if (inet_pton(AF_INET, "127.0.0.1", &serv_addr.sin_addr) <= 0) {
    std::cout << "Invalid Address / Address Not Supported" << std::endl;
  }
  if (connect(sock, (struct sockaddr *)&serv_addr, sizeof(serv_addr)) < 0) {
    std::cout << "Connection Failed" << std::endl;
  }
  int status = fcntl(sock, F_SETFL, fcntl(sock, F_GETFL, 0) | O_NONBLOCK);
  if (status == -1) {
    std::cout << "Failed to Make the Socket Non-Blocking" << std::endl;
  }
}
double VelocityController::MovingAverage(std::array<double, 100>& buffer, double input) {
  double filtered_input = 0.0;
  for (int i = 0; i < 100; i++) {
    filtered_input = filtered_input + buffer[i];
  }
  filtered_input = filtered_input / 100.0;
  for (int i = 0; i < 99; i++) {
    buffer[i] = buffer[i + 1];
  }
  buffer[99] = input;
  return filtered_input;
}
franka::JointVelocities VelocityController::operator()(const franka::RobotState& robot_state,franka::Duration period) {
  time += period.toSec();
  std::array<double, 7> joint_position = robot_state.q;
  std::array<double, 7> joint_velocity = robot_state.dq;
  std::array<double, 7> applied_torque = robot_state.tau_ext_hat_filtered;
  std::array<double, 42> jacobian = modelPtr->zeroJacobian(franka::Frame::kEndEffector, robot_state);
  double send_rate = robot_state.control_command_success_rate;
  std::string state = "s,";
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(joint_position[i]));
    state.append(",");
  }
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(joint_velocity[i]));
    state.append(",");
  }
  for (int i = 0; i < 7; i++) {
    state.append(std::to_string(applied_torque[i]));
    state.append(",");
  }
  for (int i = 0; i < 42; i++) {
    state.append(std::to_string(jacobian[i]));
    state.append(",");
  }
  char cstr[state.size() + 1];
  std::copy(state.begin(), state.end(), cstr);
  cstr[state.size()] = '\0';
  if (steps % 50 < 1) {
    valread = read(sock, buffer, 200);
    send(sock, cstr, strlen(cstr), 0);
    if (valread > 0) {
      std::stringstream ss(buffer);
      bool first = false;
      while (not first) {
        std::string substr;
        getline(ss, substr, ',');
        if ( substr[0] == 's') {
          first = true;
        }
      }
      for (int i = 0; i < 7; i++) {
        std::string substr;
        getline(ss, substr, ',');
        double term = std::stod(substr);
        control_input[i] = term;
      }
    } else {
      for (int i = 0; i < 7; i++) {
        control_input[i] = 0.0;
      }
    }
  }
  double qdot1 = MovingAverage(buffer1, control_input[0]);
  double qdot2 = MovingAverage(buffer2, control_input[1]);
  double qdot3 = MovingAverage(buffer3, control_input[2]);
  double qdot4 = MovingAverage(buffer4, control_input[3]);
  double qdot5 = MovingAverage(buffer5, control_input[4]);
  double qdot6 = MovingAverage(buffer6, control_input[5]);
  double qdot7 = MovingAverage(buffer7, control_input[6]);
  double comm_rate = MovingAverage(bufferRate, send_rate);
  qdot = {{qdot1, qdot2, qdot3, qdot4, qdot5, qdot6, qdot7}};
  franka::JointVelocities velocity(qdot);
  steps = steps + 1;
  if (steps % 1000 < 1) {
    std::cout << "control_command_success_rate: " << comm_rate << std::endl;
  }
  return velocity;
}
int main() {
  int port = 8080;
  try {
    franka::Robot robot("172.16.0.2");
    franka::Model model = robot.loadModel();
    robot.automaticErrorRecovery();
    VelocityController motion_generator(model, port);
    robot.setCollisionBehavior(
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}},
        {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}}, {{80.0, 80.0, 80.0, 80.0, 80.0, 80.0}});
    robot.setJointImpedance({{3000, 3000, 3000, 2500, 2500, 2000, 2000}});
    robot.control(motion_generator);
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}
