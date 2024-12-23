/*******************************************************************************************
*
*   raylib - classic game: asteroids
*
*   Sample game developed by Ian Eito, Albert Martos and Ramon Santamaria
*
*   This game has been created using raylib v1.3 (www.raylib.com)
*   raylib is licensed under an unmodified zlib/libpng license (View raylib.h for details)
*
*   Copyright (c) 2015 Ramon Santamaria (@raysan5)
*
********************************************************************************************/

#include "raylib.h"
#include "comms.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>



#include <math.h>

#if defined(PLATFORM_WEB)
    #include <emscripten/emscripten.h>
#endif

//----------------------------------------------------------------------------------
// Some Defines
//----------------------------------------------------------------------------------
#define PLAYER_BASE_SIZE    20.0f
#define PLAYER_SPEED        6.0f
#define PLAYER_MAX_SHOOTS   10

#define METEORS_SPEED       2
#define MAX_BIG_METEORS     4
#define MAX_MEDIUM_METEORS  8
#define MAX_SMALL_METEORS   16

#define TIME_PENALTY       -1
#define HIT_REWARD         500
#define MISS_PENALTY       -1
#define DEATH_PENALTY      10000
#define VICTORY_REWARD     10000

//----------------------------------------------------------------------------------
// Types and Structures Definition
//----------------------------------------------------------------------------------
typedef struct Player {
    Vector2 position;
    Vector2 speed;
    float acceleration;
    float rotation;
    Vector3 collider;
    Color color;
} Player;

typedef struct Shoot {
    Vector2 position;
    Vector2 speed;
    float radius;
    float rotation;
    int lifeSpawn;
    bool active;
    Color color;
} Shoot;

typedef struct Meteor {
    Vector2 position;
    Vector2 speed;
    float radius;
    bool active;
    Color color;
} Meteor;

//------------------------------------------------------------------------------------
// Global Variables Declaration
//------------------------------------------------------------------------------------
static const int screenWidth = 800;
static const int screenHeight = 450;

static bool gameOver = false;
static bool paus = false;
static bool victory = false;

// NOTE: Defined triangle is isosceles with common angles of 70 degrees.
static float shipHeight = 0.0f;

static Player player = { 0 };
static Shoot shoot[PLAYER_MAX_SHOOTS] = { 0 };
static Meteor bigMeteor[MAX_BIG_METEORS] = { 0 };
static Meteor mediumMeteor[MAX_MEDIUM_METEORS] = { 0 };
static Meteor smallMeteor[MAX_SMALL_METEORS] = { 0 };

static int destroyedMeteorsCount = 0;

static int midMeteorsCount = 0;
static int smallMeteorsCount = 0;
static int bigMeteorsCount = 0;
static int shotCount = 0;
static bool isFirstFrame = true;

int sock = 0;
int reward = 0;
int total_reward = 0;
int action = -1;

char* ACTIONS[] = {
    "FORWARD",
    "BACKWARD",
    "LEFT",
    "RIGHT",
    "SHOOT",
    "LSHOOT",
    "RSHOOT",
    "FSHOOT",
    "BSHOOT",
    "NOOP",
    "RESET"
};

char* LAST_10_ACTIONS[10] = {""};

//------------------------------------------------------------------------------------
// Module Functions Declaration (local)
//------------------------------------------------------------------------------------
static void InitGame(void);         // Initialize game
static void UpdateGame(void);       // Update game (one frame)
static void DrawGame(void);         // Draw game (one frame)
static void UnloadGame(void);       // Unload game
static void UpdateDrawFrame(void);  // Update and Draw (one frame)
int SendData(void);
//------------------------------------------------------------------------------------
// Program main entry point
//------------------------------------------------------------------------------------
int main(void)
{
    // Initialization (Note windowTitle is unused on Android)
    //---------------------------------------------------------
    sock = create_socket();
    if (connect_to_server(sock, SERVER_IP, SERVER_PORT) < 0) {
        close(sock);
        printf("Error connecting to server\n");
        return 1;
    }
    InitWindow(screenWidth, screenHeight, "classic game: asteroids");

    InitGame();

#if defined(PLATFORM_WEB)
    emscripten_set_main_loop(UpdateDrawFrame, 60, 1);
#else
    SetTargetFPS(1000);
    //--------------------------------------------------------------------------------------

    // Main game loop
    while (!WindowShouldClose())    // Detect window close button or ESC key
    {
        // Update and Draw
        //----------------------------------------------------------------------------------
        UpdateDrawFrame();
        //----------------------------------------------------------------------------------
    }
#endif
    // De-Initialization
    //--------------------------------------------------------------------------------------
    UnloadGame();         // Unload loaded data (textures, sounds, models...)

    CloseWindow();        // Close window and OpenGL context
    //--------------------------------------------------------------------------------------

    return 0;
}

//------------------------------------------------------------------------------------
// Module Functions Definitions (local)
//------------------------------------------------------------------------------------

// Initialize game variables
void InitGame(void)
{
    int posx, posy;
    int velx, vely;
    bool correctRange = false;
    victory = false;
    paus = false;
    total_reward = 0;

    shipHeight = (PLAYER_BASE_SIZE/2)/tanf(20*DEG2RAD);

    // Initialization player
    player.position = (Vector2){screenWidth/2, screenHeight/2 - shipHeight/2};
    player.speed = (Vector2){0, 0};
    player.acceleration = 0;
    player.rotation = 0;
    player.collider = (Vector3){player.position.x + sin(player.rotation*DEG2RAD)*(shipHeight/2.5f), player.position.y - cos(player.rotation*DEG2RAD)*(shipHeight/2.5f), 12};
    player.color = LIGHTGRAY;

    destroyedMeteorsCount = 0;
    // Initialization shoot
    for (int i = 0; i < PLAYER_MAX_SHOOTS; i++)
    {
        shoot[i].position = (Vector2){0, 0};
        shoot[i].speed = (Vector2){0, 0};
        shoot[i].radius = 2;
        shoot[i].active = false;
        shoot[i].lifeSpawn = 0;
        shoot[i].color = WHITE;
    }

    for (int i = 0; i < MAX_BIG_METEORS; i++)
    {
        posx = GetRandomValue(0, screenWidth);

        while (!correctRange)
        {
            if (posx > screenWidth/2 - 150 && posx < screenWidth/2 + 150) posx = GetRandomValue(0, screenWidth);
            else correctRange = true;
        }

        correctRange = false;

        posy = GetRandomValue(0, screenHeight);

        while (!correctRange)
        {
            if (posy > screenHeight/2 - 150 && posy < screenHeight/2 + 150)  posy = GetRandomValue(0, screenHeight);
            else correctRange = true;
        }

        bigMeteor[i].position = (Vector2){posx, posy};

        correctRange = false;
        velx = GetRandomValue(-METEORS_SPEED, METEORS_SPEED);
        vely = GetRandomValue(-METEORS_SPEED, METEORS_SPEED);

        while (!correctRange)
        {
            if (velx == 0 && vely == 0)
            {
                velx = GetRandomValue(-METEORS_SPEED, METEORS_SPEED);
                vely = GetRandomValue(-METEORS_SPEED, METEORS_SPEED);
            }
            else correctRange = true;
        }

        bigMeteor[i].speed = (Vector2){velx, vely};
        bigMeteor[i].radius = 40;
        bigMeteor[i].active = true;
        bigMeteor[i].color = BLUE;
    }

    for (int i = 0; i < MAX_MEDIUM_METEORS; i++)
    {
        mediumMeteor[i].position = (Vector2){-100, -100};
        mediumMeteor[i].speed = (Vector2){0,0};
        mediumMeteor[i].radius = 20;
        mediumMeteor[i].active = false;
        mediumMeteor[i].color = BLUE;
    }

    for (int i = 0; i < MAX_SMALL_METEORS; i++)
    {
        smallMeteor[i].position = (Vector2){-100, -100};
        smallMeteor[i].speed = (Vector2){0,0};
        smallMeteor[i].radius = 10;
        smallMeteor[i].active = false;
        smallMeteor[i].color = BLUE;
    }
}

int SendData(void) {
    char response[BUFFER_SIZE];
    char message[BUFFER_SIZE];
    int chars = 0;
    char s_meteors[BUFFER_SIZE];
    
    // Start JSON object
    chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, "{\"shotsCount\":%d,\"player\":{\"x\":%2.5f,\"y\":%2.5f,\"rotation\":%2.5f},\"meteors\":[", shotCount, player.position.x, player.position.y, player.rotation);

    // Add small meteor positions and speeds to JSON array
    for (int i = 0; i < smallMeteorsCount; i++) {
        if (i > 0) {
            chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, ",");
        }
        chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, "{\"type\":\"small\",\"x\":%2.5f,\"y\":%2.5f,\"speed_x\":%2.5f,\"speed_y\":%2.5f}", smallMeteor[i].position.x, smallMeteor[i].position.y, smallMeteor[i].speed.x, smallMeteor[i].speed.y);
    }

    // Add medium meteor positions and speeds to JSON array
    for (int i = 0; i < midMeteorsCount; i++) {
        if (i > 0 || smallMeteorsCount > 0) {
            chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, ",");
        }
        chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, "{\"type\":\"medium\",\"x\":%2.5f,\"y\":%2.5f,\"speed_x\":%2.5f,\"speed_y\":%2.5f}", mediumMeteor[i].position.x, mediumMeteor[i].position.y, mediumMeteor[i].speed.x, mediumMeteor[i].speed.y);
    }

    // Add big meteor positions and speeds to JSON array
    for (int i = 0; i < bigMeteorsCount; i++) {
        if (i > 0 || smallMeteorsCount > 0 || midMeteorsCount > 0) {
            chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, ",");
        }
        chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, "{\"type\":\"big\",\"x\":%2.5f,\"y\":%2.5f,\"speed_x\":%2.5f,\"speed_y\":%2.5f}", bigMeteor[i].position.x, bigMeteor[i].position.y, bigMeteor[i].speed.x, bigMeteor[i].speed.y);
    }

    // End JSON array and start shots array
    // 0 -> game not over, 1 -> victory, 2 -> defeat
    int game_state = 0;
    if(gameOver){
        if(victory) game_state = 1;
        else game_state = 2;
    }
    chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, "],\"reward\":%d,\"game_over\":%d,\"shots\":[", reward, game_state);

    // Add player shots positions and rotations to JSON array
    for (int i = 0; i < shotCount; i++) {
        if (i > 0) {
            chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, ",");
        }
        chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, "{\"x\":%2.5f,\"y\":%2.5f,\"rotation\":%2.5f}", shoot[i].position.x, shoot[i].position.y, shoot[i].rotation);
    }

    // End JSON array and object
    chars += snprintf(s_meteors + chars, BUFFER_SIZE - chars, "]}");

    // Ensure null-termination
    s_meteors[BUFFER_SIZE - 1] = '\0';
    strncpy(message, s_meteors, BUFFER_SIZE - 1);
    message[BUFFER_SIZE - 1] = '\0';

    // printf("Sending message: %s\n", message);
    // printf("Length of message: %ld\n", strlen(message));
    if(strlen(message) > BUFFER_SIZE - BUFFER_SIZE/10) {
        printf("Message too long\n");
        close(sock);
        exit(1);
    }

    // Send a message and receive a response
    if (send_message(sock, message, response, BUFFER_SIZE) < 0) {
        printf("error sending message\n");
        close(sock);
        exit(1);
    }

/*
ACTIONS = {
    0: "FORWARD",
    1: "BACKWARD",
    2: "LEFT",
    3: "RIGHT",
    4: "SHOOT",
    5: "LSHOOT",
    6: "RSHOOT",
    7: "FSHOOT",
    8: "BSHOOT",
    9: "NOOP"
}
*/
    if(strcmp(response, "RESET") == 0) {
        action = 10;
        // printf("Resetting game\n");
        // InitGame();
    }
    if(strcmp(response, "FORWARD") == 0){
        action = 0;
    }
    if(strcmp(response, "BACKWARD") == 0){
        action = 1;
    }
    if(strcmp(response, "LEFT") == 0){
        action = 2;
    }
    if(strcmp(response, "RIGHT") == 0){
        action = 3;
    }
    if(strcmp(response, "SHOOT") == 0){
        action = 4;
    }
    if(strcmp(response, "LSHOOT") == 0){
        action = 5;
    }
    if(strcmp(response, "RSHOOT") == 0){
        action = 6;
    }
    if(strcmp(response, "FSHOOT") == 0){
        action = 7;
    }
    if(strcmp(response, "BSHOOT") == 0){
        action = 8;
    }
    if(strcmp(response, "NOOP") == 0){
        action = 9;
    }

    // push action to last 10 actions
    for (int i = 0; i < 9; i++) {
        LAST_10_ACTIONS[i] = LAST_10_ACTIONS[i + 1];
    }
    LAST_10_ACTIONS[9] = ACTIONS[action];
    return 0;
}

// Update game (one frame)
void UpdateGame(void){
    reward = TIME_PENALTY;
    
    if (!gameOver)
    {
        if (IsKeyPressed('P')) paus = !paus;

        if (!paus)
        {
            if(IsKeyPressed('M')) InitGame();
            // Player logic: rotation
            // if (IsKeyDown(KEY_LEFT)) player.rotation -= 5;
            // if (IsKeyDown(KEY_RIGHT)) player.rotation += 5;
            if(action == 2 || action == 5) player.rotation -= 5;
            if(action == 3 || action == 6) player.rotation += 5;

            // Player logic: speed
            player.speed.x = sin(player.rotation*DEG2RAD)*PLAYER_SPEED;
            player.speed.y = cos(player.rotation*DEG2RAD)*PLAYER_SPEED;

            // Player logic: acceleration
            // if (IsKeyDown(KEY_UP))
            if(action == 0 || action == 7)
            {
                if (player.acceleration < 1) player.acceleration += 0.04f;
            }
            else
            {
                if (player.acceleration > 0) player.acceleration -= 0.02f;
                else if (player.acceleration < 0) player.acceleration = 0;
            }
            if (action == 1 || action == 8)
            {
                if (player.acceleration > 0) player.acceleration -= 0.04f;
                else if (player.acceleration < 0) player.acceleration = 0;
            }

            // Player logic: movement
            player.position.x += (player.speed.x*player.acceleration);
            player.position.y -= (player.speed.y*player.acceleration);

            // Collision logic: player vs walls
            if (player.position.x > screenWidth + shipHeight) player.position.x = -(shipHeight);
            else if (player.position.x < -(shipHeight)) player.position.x = screenWidth + shipHeight;
            if (player.position.y > (screenHeight + shipHeight)) player.position.y = -(shipHeight);
            else if (player.position.y < -(shipHeight)) player.position.y = screenHeight + shipHeight;

            // Player shoot logic
            if (action == 4 || action == 5 || action == 6 || action == 7 || action == 8)
            {
                reward += MISS_PENALTY;
                for (int i = 0; i < PLAYER_MAX_SHOOTS; i++)
                {
                    if (!shoot[i].active)
                    {
                        shoot[i].position = (Vector2){ player.position.x + sin(player.rotation*DEG2RAD)*(shipHeight), player.position.y - cos(player.rotation*DEG2RAD)*(shipHeight) };
                        shoot[i].active = true;
                        shoot[i].speed.x = 1.5*sin(player.rotation*DEG2RAD)*PLAYER_SPEED;
                        shoot[i].speed.y = 1.5*cos(player.rotation*DEG2RAD)*PLAYER_SPEED;
                        shoot[i].rotation = player.rotation;
                        shotCount++;
                        break;
                    }
                }
            }

            // Shoot life timer
            for (int i = 0; i < PLAYER_MAX_SHOOTS; i++)
            {
                if (shoot[i].active) shoot[i].lifeSpawn++;
            }

            // Shot logic
            for (int i = 0; i < PLAYER_MAX_SHOOTS; i++)
            {
                if (shoot[i].active)
                {
                    // Movement
                    shoot[i].position.x += shoot[i].speed.x;
                    shoot[i].position.y -= shoot[i].speed.y;

                    // Collision logic: shoot vs walls
                    if  (shoot[i].position.x > screenWidth + shoot[i].radius)
                    {
                        shoot[i].active = false;
                        shoot[i].lifeSpawn = 0;
                    }
                    else if (shoot[i].position.x < 0 - shoot[i].radius)
                    {
                        shoot[i].active = false;
                        shoot[i].lifeSpawn = 0;
                    }
                    if (shoot[i].position.y > screenHeight + shoot[i].radius)
                    {
                        shoot[i].active = false;
                        shoot[i].lifeSpawn = 0;
                    }
                    else if (shoot[i].position.y < 0 - shoot[i].radius)
                    {
                        shoot[i].active = false;
                        shoot[i].lifeSpawn = 0;
                    }

                    // Life of shoot
                    if (shoot[i].lifeSpawn >= 60)
                    {
                        shoot[i].position = (Vector2){0, 0};
                        shoot[i].speed = (Vector2){0, 0};
                        shoot[i].lifeSpawn = 0;
                        shoot[i].active = false;
                    }
                }
            }

            // Collision logic: player vs meteors
            player.collider = (Vector3){player.position.x + sin(player.rotation*DEG2RAD)*(shipHeight/2.5f), player.position.y - cos(player.rotation*DEG2RAD)*(shipHeight/2.5f), 12};

            for (int a = 0; a < MAX_BIG_METEORS; a++)
            {
                if (CheckCollisionCircles((Vector2){player.collider.x, player.collider.y}, player.collider.z, bigMeteor[a].position, bigMeteor[a].radius) && bigMeteor[a].active) gameOver = true;
            }

            for (int a = 0; a < MAX_MEDIUM_METEORS; a++)
            {
                if (CheckCollisionCircles((Vector2){player.collider.x, player.collider.y}, player.collider.z, mediumMeteor[a].position, mediumMeteor[a].radius) && mediumMeteor[a].active) gameOver = true;
            }

            for (int a = 0; a < MAX_SMALL_METEORS; a++)
            {
                if (CheckCollisionCircles((Vector2){player.collider.x, player.collider.y}, player.collider.z, smallMeteor[a].position, smallMeteor[a].radius) && smallMeteor[a].active) gameOver = true;
            }

            if(gameOver) reward -= DEATH_PENALTY;

            // Meteors logic: big meteors
            for (int i = 0; i < MAX_BIG_METEORS; i++)
            {
                if (bigMeteor[i].active)
                {
                    // Movement
                    bigMeteor[i].position.x += bigMeteor[i].speed.x;
                    bigMeteor[i].position.y += bigMeteor[i].speed.y;

                    // Collision logic: meteor vs wall
                    if  (bigMeteor[i].position.x > screenWidth + bigMeteor[i].radius) bigMeteor[i].position.x = -(bigMeteor[i].radius);
                    else if (bigMeteor[i].position.x < 0 - bigMeteor[i].radius) bigMeteor[i].position.x = screenWidth + bigMeteor[i].radius;
                    if (bigMeteor[i].position.y > screenHeight + bigMeteor[i].radius) bigMeteor[i].position.y = -(bigMeteor[i].radius);
                    else if (bigMeteor[i].position.y < 0 - bigMeteor[i].radius) bigMeteor[i].position.y = screenHeight + bigMeteor[i].radius;
                }
            }

            // Meteors logic: medium meteors
            for (int i = 0; i < MAX_MEDIUM_METEORS; i++)
            {

                if (mediumMeteor[i].active)
                {
                    // Movement
                    mediumMeteor[i].position.x += mediumMeteor[i].speed.x;
                    mediumMeteor[i].position.y += mediumMeteor[i].speed.y;

                    // Collision logic: meteor vs wall
                    if  (mediumMeteor[i].position.x > screenWidth + mediumMeteor[i].radius) mediumMeteor[i].position.x = -(mediumMeteor[i].radius);
                    else if (mediumMeteor[i].position.x < 0 - mediumMeteor[i].radius) mediumMeteor[i].position.x = screenWidth + mediumMeteor[i].radius;
                    if (mediumMeteor[i].position.y > screenHeight + mediumMeteor[i].radius) mediumMeteor[i].position.y = -(mediumMeteor[i].radius);
                    else if (mediumMeteor[i].position.y < 0 - mediumMeteor[i].radius) mediumMeteor[i].position.y = screenHeight + mediumMeteor[i].radius;
                }
            }

            // Meteors logic: small meteors
            for (int i = 0; i < MAX_SMALL_METEORS; i++)
            {
                if (smallMeteor[i].active)
                {
                    // Movement
                    smallMeteor[i].position.x += smallMeteor[i].speed.x;
                    smallMeteor[i].position.y += smallMeteor[i].speed.y;

                    // Collision logic: meteor vs wall
                    if  (smallMeteor[i].position.x > screenWidth + smallMeteor[i].radius) smallMeteor[i].position.x = -(smallMeteor[i].radius);
                    else if (smallMeteor[i].position.x < 0 - smallMeteor[i].radius) smallMeteor[i].position.x = screenWidth + smallMeteor[i].radius;
                    if (smallMeteor[i].position.y > screenHeight + smallMeteor[i].radius) smallMeteor[i].position.y = -(smallMeteor[i].radius);
                    else if (smallMeteor[i].position.y < 0 - smallMeteor[i].radius) smallMeteor[i].position.y = screenHeight + smallMeteor[i].radius;
                }
            }

            // Collision logic: player-shoots vs meteors
            for (int i = 0; i < PLAYER_MAX_SHOOTS; i++)
            {
                if ((shoot[i].active))
                {
                    for (int a = 0; a < MAX_BIG_METEORS; a++)
                    {
                        if (bigMeteor[a].active && CheckCollisionCircles(shoot[i].position, shoot[i].radius, bigMeteor[a].position, bigMeteor[a].radius))
                        {
                            reward += HIT_REWARD;
                            shoot[i].active = false;
                            shoot[i].lifeSpawn = 0;
                            bigMeteor[a].active = false;
                            destroyedMeteorsCount++;

                            for (int j = 0; j < 2; j ++)
                            {
                                if (midMeteorsCount%2 == 0)
                                {
                                    mediumMeteor[midMeteorsCount].position = (Vector2){bigMeteor[a].position.x, bigMeteor[a].position.y};
                                    mediumMeteor[midMeteorsCount].speed = (Vector2){cos(shoot[i].rotation*DEG2RAD)*METEORS_SPEED*-1, sin(shoot[i].rotation*DEG2RAD)*METEORS_SPEED*-1};
                                }
                                else
                                {
                                    mediumMeteor[midMeteorsCount].position = (Vector2){bigMeteor[a].position.x, bigMeteor[a].position.y};
                                    mediumMeteor[midMeteorsCount].speed = (Vector2){cos(shoot[i].rotation*DEG2RAD)*METEORS_SPEED, sin(shoot[i].rotation*DEG2RAD)*METEORS_SPEED};
                                }

                                mediumMeteor[midMeteorsCount].active = true;
                                midMeteorsCount ++;
                            }
                            //bigMeteor[a].position = (Vector2){-100, -100};
                            bigMeteor[a].color = RED;
                            a = MAX_BIG_METEORS;
                        }
                    }

                    for (int b = 0; b < MAX_MEDIUM_METEORS; b++)
                    {
                        if (mediumMeteor[b].active && CheckCollisionCircles(shoot[i].position, shoot[i].radius, mediumMeteor[b].position, mediumMeteor[b].radius))
                        {
                            reward += HIT_REWARD;
                            shoot[i].active = false;
                            shoot[i].lifeSpawn = 0;
                            mediumMeteor[b].active = false;
                            destroyedMeteorsCount++;

                            for (int j = 0; j < 2; j ++)
                            {
                                 if (smallMeteorsCount%2 == 0)
                                {
                                    smallMeteor[smallMeteorsCount].position = (Vector2){mediumMeteor[b].position.x, mediumMeteor[b].position.y};
                                    smallMeteor[smallMeteorsCount].speed = (Vector2){cos(shoot[i].rotation*DEG2RAD)*METEORS_SPEED*-1, sin(shoot[i].rotation*DEG2RAD)*METEORS_SPEED*-1};
                                }
                                else
                                {
                                    smallMeteor[smallMeteorsCount].position = (Vector2){mediumMeteor[b].position.x, mediumMeteor[b].position.y};
                                    smallMeteor[smallMeteorsCount].speed = (Vector2){cos(shoot[i].rotation*DEG2RAD)*METEORS_SPEED, sin(shoot[i].rotation*DEG2RAD)*METEORS_SPEED};
                                }

                                smallMeteor[smallMeteorsCount].active = true;
                                smallMeteorsCount ++;
                            }
                            mediumMeteor[b].color = GREEN;
                            b = MAX_MEDIUM_METEORS;
                        }
                    }

                    for (int c = 0; c < MAX_SMALL_METEORS; c++)
                    {
                        if (smallMeteor[c].active && CheckCollisionCircles(shoot[i].position, shoot[i].radius, smallMeteor[c].position, smallMeteor[c].radius))
                        {
                            shoot[i].active = false;
                            shoot[i].lifeSpawn = 0;
                            smallMeteor[c].active = false;
                            destroyedMeteorsCount++;
                            reward += HIT_REWARD;
                            smallMeteor[c].color = YELLOW;
                            c = MAX_SMALL_METEORS;
                        }
                    }
                }
            }
        }

        int newBigMeteorCount = 0;
        for (int i = 0; i < MAX_BIG_METEORS; i++){
            if(bigMeteor[i].active){
                newBigMeteorCount++;
            }
        }
        bigMeteorsCount = newBigMeteorCount;

        int newMediumMeteorCount = 0;
        for (int i = 0; i < MAX_MEDIUM_METEORS; i++){
            if(mediumMeteor[i].active){
                newMediumMeteorCount++;
            }
        }
        midMeteorsCount = newMediumMeteorCount;

        int newSmallMeteorCount = 0;
        for (int i = 0; i < MAX_SMALL_METEORS; i++){
            if(smallMeteor[i].active){
                newSmallMeteorCount++;
            }
        }
        smallMeteorsCount = newSmallMeteorCount;

        int newShotCount = 0;
        for (int i = 0; i < PLAYER_MAX_SHOOTS; i++){
            if(shoot[i].active){
                newShotCount++;
            }
        }
        shotCount = newShotCount;


        if (0 == bigMeteorsCount + midMeteorsCount + smallMeteorsCount) victory = true;
        if (victory) {
            reward += VICTORY_REWARD;
            gameOver = true;
        }
        total_reward += reward;
    }
    else
    {
        if (IsKeyPressed(KEY_ENTER))
        {
            InitGame();
            gameOver = false;
        }
    }
    if (isFirstFrame) isFirstFrame = false; else SendData(); 
    if(action == 10) {
        InitGame();
        gameOver = false;
    }
}

// Draw game (one frame)
void DrawGame(void)
{
    BeginDrawing();

        ClearBackground(RAYWHITE);

        if (!gameOver)
        {
            // Draw spaceship
            Vector2 v1 = { player.position.x + sinf(player.rotation*DEG2RAD)*(shipHeight), player.position.y - cosf(player.rotation*DEG2RAD)*(shipHeight) };
            Vector2 v2 = { player.position.x - cosf(player.rotation*DEG2RAD)*(PLAYER_BASE_SIZE/2), player.position.y - sinf(player.rotation*DEG2RAD)*(PLAYER_BASE_SIZE/2) };
            Vector2 v3 = { player.position.x + cosf(player.rotation*DEG2RAD)*(PLAYER_BASE_SIZE/2), player.position.y + sinf(player.rotation*DEG2RAD)*(PLAYER_BASE_SIZE/2) };
            DrawTriangle(v1, v2, v3, MAROON);

            char reward_text[32];
            sprintf(reward_text, "reward: %d", total_reward);
            reward_text[31] = '\0';
            DrawText(reward_text, 10, 10, 20, BLACK);

            char meteor_text[32];
            sprintf(meteor_text, "meteors: %d", MAX_BIG_METEORS + MAX_MEDIUM_METEORS + MAX_SMALL_METEORS - destroyedMeteorsCount);
            meteor_text[31] = '\0';
            DrawText(meteor_text, 10, 40, 20, BLACK);


            char meteor_text2[128];
            sprintf(meteor_text2, "big: %d, medium: %d, small: %d", bigMeteorsCount, midMeteorsCount, smallMeteorsCount);
            meteor_text2[127] = '\0';
            DrawText(meteor_text2, 160, 10, 20, BLACK);

            // draw last 10 actions
            for (int i = 0; i < 10; i++) {
                DrawText(LAST_10_ACTIONS[i], 10, 70 + 30 * i, 20, BLACK);
            }




            // Draw meteors
            for (int i = 0; i < MAX_BIG_METEORS; i++)
            {
                if (bigMeteor[i].active) DrawCircleV(bigMeteor[i].position, bigMeteor[i].radius, DARKGRAY);
                else DrawCircleV(bigMeteor[i].position, bigMeteor[i].radius, Fade(LIGHTGRAY, 0.3f));
            }

            for (int i = 0; i < MAX_MEDIUM_METEORS; i++)
            {
                if (mediumMeteor[i].active) DrawCircleV(mediumMeteor[i].position, mediumMeteor[i].radius, GRAY);
                else DrawCircleV(mediumMeteor[i].position, mediumMeteor[i].radius, Fade(LIGHTGRAY, 0.3f));
            }

            for (int i = 0; i < MAX_SMALL_METEORS; i++)
            {
                if (smallMeteor[i].active) DrawCircleV(smallMeteor[i].position, smallMeteor[i].radius, GRAY);
                else DrawCircleV(smallMeteor[i].position, smallMeteor[i].radius, Fade(LIGHTGRAY, 0.3f));
            }

            // Draw shoot
            for (int i = 0; i < PLAYER_MAX_SHOOTS; i++)
            {
                if (shoot[i].active) DrawCircleV(shoot[i].position, shoot[i].radius, BLACK);
            }

            if (victory) DrawText("VICTORY", screenWidth/2 - MeasureText("VICTORY", 20)/2, screenHeight/2, 20, LIGHTGRAY);

            if (paus) DrawText("GAME PAUSED", screenWidth/2 - MeasureText("GAME PAUSED", 40)/2, screenHeight/2 - 40, 40, GRAY);
        }
        else DrawText("PRESS [ENTER] TO PLAY AGAIN", GetScreenWidth()/2 - MeasureText("PRESS [ENTER] TO PLAY AGAIN", 20)/2, GetScreenHeight()/2 - 50, 20, GRAY);

    EndDrawing();
}

// Unload game variables
void UnloadGame(void)
{
    // Close the socket
    close(sock);
}

// Update and Draw (one frame)
void UpdateDrawFrame(void)
{
    UpdateGame();
    DrawGame();
}
