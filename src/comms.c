#include "comms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <arpa/inet.h>
#include <unistd.h>

// Create a socket
int create_socket() {
    int sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("Socket creation error");
        return -1;
    }
    return sock;
}

// Connect to the server
int connect_to_server(int sock, const char *server_ip, int server_port) {
    struct sockaddr_in server_address;

    server_address.sin_family = AF_INET;
    server_address.sin_port = htons(server_port);

    // Convert IP address to binary
    if (inet_pton(AF_INET, server_ip, &server_address.sin_addr) <= 0) {
        perror("Invalid address/Address not supported");
        return -1;
    }

    if (connect(sock, (struct sockaddr *)&server_address, sizeof(server_address)) < 0) {
        perror("Connection failed");
        return -1;
    }
    return 0;
}

// Send message to the server and receive a response
int send_message(int sock, const char *message, char *response, size_t response_size) {
    if (send(sock, message, strlen(message), 0) < 0) {
        perror("Send failed");
        return -1;
    }

    ssize_t bytes_received = recv(sock, response, response_size - 1, 0);
    if (bytes_received < 0) {
        perror("Receive failed");
        return -1;
    }

    response[bytes_received] = '\0';  // Null terminate received data
    return 0;
}
