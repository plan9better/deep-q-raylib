#ifndef TCP_CLIENT_H
#define TCP_CLIENT_H

#include <stddef.h>

// Constants
#define SERVER_IP "127.0.0.1"
// #define SERVER_PORT 12345
#define SERVER_PORT 65432

#define BUFFER_SIZE 4096

// Function prototypes
int create_socket();
int connect_to_server(int sock, const char *server_ip, int server_port);
int send_message(int sock, const char *message, char *response, size_t response_size);

#endif // TCP_CLIENT_H
