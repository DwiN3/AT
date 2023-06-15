#pragma once
#ifndef CUBE_H
#define CUBE_H


#include <glad/glad.h>
#include <GLFW/glfw3.h>
//#include <C:/LIB/stb/stb_image.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <C:/LIB/assets/camera.h>
#include <C:/LIB/assets/shader.h>
#include <iostream>
#include <glm/gtc/type_ptr.hpp>
#include <random>
#include <stdlib.h>
#include <chrono>
#include <thread>
#include <filesystem>
#include <fstream>
#include <string>
#include <iomanip>
#include <algorithm>



using namespace std;
using namespace std::experimental::filesystem;



/// <summary>
/// Szerokoœæ okna.
/// </summary>
extern const unsigned int SCR_WIDTH;
/// <summary>
/// Wysokoœæ okna.
/// </summary>
extern const unsigned int SCR_HEIGHT;

/// <summary>
/// Ustawienia pozycji początkowej kamery.
/// </summary>
extern Camera camera;

/// <summary>
/// Poprzednia pozycja X, używana do śledzenia ruchu myszy.
/// </summary>
extern float lastX;

/// <summary>
/// Poprzednia pozycja Y, używana do śledzenia ruchu myszy.
/// </summary>
extern float lastY;

/// <summary>
/// Dodaj obsługę myszki
/// </summary>
extern bool firstMouse;

/// <summary>
/// Zmienna ustawiająca wartość pomiędzy klatkami.
/// </summary>
extern float deltaTime;
/// <summary>
/// Zmienna ustawiająca wartość ostaniej klatki.
/// </summary>
extern float lastFrame;

/// <summary>
/// Zmienna przyjmująca wartość startu timera.
/// </summary>
extern chrono::time_point<chrono::high_resolution_clock> start_timer;
/// <summary>
/// Zmienna przyjmująca wartość zatrzymania timera.
/// </summary>
extern chrono::time_point<chrono::high_resolution_clock> end_timer;
/// <summary>
/// Zmienna przechowywująca czas ułożenia kostki.
/// </summary>
extern chrono::duration<double> duration;

/// <summary>
/// Zmienna przechowująca ilość rekordów.
/// </summary>
extern const int TOP_SCORES_COUNT;
/// <summary>
/// Ranking czasów ułożenia.
/// </summary>
extern double top_scores[5];

/// <summary>
/// Wybór koloru odpowiedniego dla danego użytkownika.
/// </summary>
extern int color;

/// <summary>
/// Zmiena przechowująca stan układania kostki.
/// </summary>
extern bool arranging;

/// <summary>
/// Zmienne przechowujące wykonane ruchy.
/// </summary>
extern int count_moves;
/// <summary>
/// Zmienne przechowujące wykonane ruchy poprzez mieszanie.
/// </summary>
extern int random_moves;

/// <summary>
/// Zmienne przechowujące podpowiedź.
/// </summary>
extern string solve;
/// <summary>
/// Zmienne przechowujące stan użycia podpowiedzi.
/// </summary>
extern bool isSolved;

/// <summary>
/// Zmienne przechowujące domyślne tekstury.
/// </summary>
extern unsigned int textureClassic1, textureClassic2, textureClassic3, textureClassic4, textureClassic5, textureClassic6;
/// <summary>
/// Zmienne przechowujące tekstury z deuteranopii.
/// </summary>
extern unsigned int textureDeuteranopia1, textureDeuteranopia2, textureDeuteranopia3, textureDeuteranopia4, textureDeuteranopia5, textureDeuteranopia6;
/// <summary>
/// Zmienne przechowujące tekstury z tritanopii.
/// </summary>
extern unsigned int textureTritanopia1, textureTritanopia2, textureTritanopia3, textureTritanopia4, textureTritanopia5, textureTritanopia6;



/// <summary>
/// Tablice wyœwietlają serce kostki rubika która zapamiętuje swój stan.
/// </summary>
extern int sideCube[27][6];

/// <summary>
/// Pozycje kostek czyli rozmieszczenie kostek aby uformowały się w jedną kostkę "Rubik's Cube.
/// </summary>
extern glm::vec3 cubePositions[27];


void framebuffer_size_callback(GLFWwindow* window, int width, int height);
void mouse_callback(GLFWwindow* window, double xpos, double ypos);
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
void processInput(GLFWwindow* window);
unsigned int loadCubemap();
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
void turn_cube_up_to_down(int which, int each, bool game);
void turn_cube_down_to_up(int which, int each, bool game);
void turn_cube_left_to_right(int which, int each, bool game);
void turn_cube_right_to_left(int which, int each, bool game);
void mix_the_cube(int mode);
void turn_cube_to_full();
void print_cube_color();
void cube_arranged(bool skip);
void set_best_scores();
void show_best_scores();
void show_options();
bool is_cube_solved();
void show_solve();
void loadTextures();
void loadData(unsigned char* data, int width, int height);
void dataTextureLoad();


#endif // CUBE_H