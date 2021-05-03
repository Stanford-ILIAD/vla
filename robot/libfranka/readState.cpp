/**
 * readState.cpp
 *
 * A Minimal Script for Getting the Robot State when Not Controlling the Robot Programmatically.
 */

#include <iostream>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <cstring>
#include <fcntl.h>

#include <franka/exception.h>
#include <franka/robot.h>

int connect2control(int PORT) {
  int sock = 0;
  struct sockaddr_in serv_addr;
  if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
    std::cout << "Socket Creation Error!" << std::endl;
  }
  serv_addr.sin_family = AF_INET;
  serv_addr.sin_port = htons(PORT);
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
  return sock;
}


void send2control(std::string state, int sock) {
  char cstr[state.size() + 1];
  std::copy(state.begin(), state.end(), cstr);
  cstr[state.size()] = '\0';
  send(sock, cstr, strlen(cstr), 0);
}


int main(int argc, char** argv) {
  if (argc != 3) {
    std::cerr << "Usage: " << std::endl
              << "IP of robot" << std::endl
              << "Port for socket channel" << std::endl
              << "example IP is [172.16.0.2]." << std::endl
              << "example Port is [8080]." << std::endl;
    return -1;
  }

  try {
    int port = std::stoi(argv[2]);

    franka::Robot robot(argv[1]);
    int sock = connect2control(port);

    while (true) {
      franka::RobotState current_state = robot.readOnce();

      std::string state = "s,";
      std::array<double, 7> joint_position = current_state.q;
      std::array<double, 7> joint_velocity = current_state.dq;
      std::array<double, 7> joint_torque = current_state.tau_ext_hat_filtered;
      for (int i = 0; i < 7; i++) {
        state.append(std::to_string(joint_position[i]));
        state.append(",");
      }
      for (int i = 0; i < 7; i++) {
        state.append(std::to_string(joint_velocity[i]));
        state.append(",");
      }
      for (int i = 0; i < 7; i++) {
        state.append(std::to_string(joint_torque[i]));
        state.append(",");
      }
      send2control(state, sock);
    }
  } catch (franka::Exception const& e) {
    std::cout << e.what() << std::endl;
    return -1;
  }
  return 0;
}
