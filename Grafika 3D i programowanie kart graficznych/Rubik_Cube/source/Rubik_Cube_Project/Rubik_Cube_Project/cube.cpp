#include "cube.h"
#define STB_IMAGE_IMPLEMENTATION
#include <C:/LIB/stb/stb_image.h>

/// <summary>
/// Szerokoœæ okna.
/// </summary>
const unsigned int SCR_WIDTH = 800;
/// <summary>
/// Wysokoœæ okna.
/// </summary>
const unsigned int SCR_HEIGHT = 600;

/// <summary>
/// Ustawienia pozycji pocz¹tkowej kamery.
/// </summary>
Camera camera = Camera(glm::vec3(0.0f, 1.0f, 8.0f));

/// <summary>
/// Poprzednia pozycja X, u¿ywana do œledzenia ruchu myszy.
/// </summary>
float lastX = SCR_WIDTH / 2.0f;

/// <summary>
/// Poprzednia pozycja Y, u¿ywana do œledzenia ruchu myszy.
/// </summary>
float lastY = SCR_HEIGHT / 2.0f;

/// <summary>
/// Dodaj obs³ugê myszki
/// </summary>
bool firstMouse = true;

/// <summary>
/// Zmienna ustawiaj¹ca wartoœæ pomiêdzy klatkami.
/// </summary>
float deltaTime = 0.0f;
/// <summary>
/// Zmienna ustawiaj¹ca wartoœæ ostaniej klatki.
/// </summary>
float lastFrame = 0.0f;

/// <summary>
/// Zmienna przyjmuj¹ca wartoœæ startu timera.
/// </summary>
chrono::time_point<chrono::high_resolution_clock> start_timer;
/// <summary>
/// Zmienna przyjmuj¹ca wartoœæ zatrzymania timera.
/// </summary>
chrono::time_point<chrono::high_resolution_clock> end_timer;
/// <summary>
/// Zmienna przechowywuj¹ca czas u³o¿enia kostki.
/// </summary>
chrono::duration<double> duration;

/// <summary>
/// Zmienna przechowuj¹ca iloœæ rekordów.
/// </summary>
const int TOP_SCORES_COUNT = 5;
/// <summary>
/// Ranking czasów u³o¿enia.
/// </summary>
double top_scores[TOP_SCORES_COUNT] = { 0.0 };

/// <summary>
/// Wybór koloru odpowiedniego dla danego u¿ytkownika.
/// </summary>
int color = 1;

/// <summary>
/// Zmiena przechowuj¹ca stan uk³adania kostki.
/// </summary>
bool arranging = false;

/// <summary>
/// Zmienne przechowuj¹ce wykonane ruchy.
/// </summary>
int count_moves = 0;
/// <summary>
/// Zmienne przechowuj¹ce wykonane ruchy poprzez mieszanie.
/// </summary>
int random_moves = 0;

/// <summary>
/// Zmienne przechowujące podpowiedzi.
/// </summary>
string solve = "";
/// <summary>
/// Zmienne przechowujące stan użycia podpowiedzi.
/// </summary>
bool isSolved = true;

/// <summary>
/// Zmienne przechowujące domyślne tekstury.
/// </summary>
unsigned int textureClassic1, textureClassic2, textureClassic3, textureClassic4, textureClassic5, textureClassic6;
/// <summary>
/// Zmienne przechowujące tekstury z deuteranopii.
/// </summary>
unsigned int textureDeuteranopia1, textureDeuteranopia2, textureDeuteranopia3, textureDeuteranopia4, textureDeuteranopia5, textureDeuteranopia6;
/// <summary>
/// Zmienne przechowujace tekstury z tritanopii.
/// </summary>
unsigned int textureTritanopia1, textureTritanopia2, textureTritanopia3, textureTritanopia4, textureTritanopia5, textureTritanopia6;

int sideCube[27][6] = {
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6},
    {1,2,3,4,5,6}
};

glm::vec3 cubePositions[] = {
    //x               y      z

     // 1 side
        glm::vec3(-1.0f,  0.0f,  -1.0f), // FRONT - LEFT - UP
        glm::vec3(0.0f,  0.0f,  -1.0f), // FRONT - CENTER UP
        glm::vec3(1.0f,   0.0f,  -1.0f),

        glm::vec3(-1.0f, -1.0f,  -1.0f),
        glm::vec3(0.0f, -1.0f,  -1.0f),
        glm::vec3(1.0f,  -1.0f,  -1.0f),

        glm::vec3(-1.0f, -2.0f,  -1.0f),
        glm::vec3(0.0f, -2.0f,  -1.0f),
        glm::vec3(1.0f,  -2.0f,  -1.0f),

        // 2 side
        glm::vec3(-1.0f,  0.0f,  -2.0f),
        glm::vec3(0.0f,  0.0f,  -2.0f),
        glm::vec3(1.0f,   0.0f,  -2.0f),

        glm::vec3(-1.0f, -1.0f,  -2.0f),
        glm::vec3(0.0f, -1.0f,  -2.0f),
        glm::vec3(1.0f,  -1.0f,  -2.0f),

        glm::vec3(-1.0f, -2.0f,  -2.0f),
        glm::vec3(0.0f, -2.0f,  -2.0f),
        glm::vec3(1.0f,  -2.0f,  -2.0f),

        // 3 side
        glm::vec3(-1.0f,  0.0f,  -3.0f),
        glm::vec3(0.0f,  0.0f,  -3.0f),
        glm::vec3(1.0f,   0.0f,  -3.0f),

        glm::vec3(-1.0f, -1.0f,  -3.0f),
        glm::vec3(0.0f, -1.0f,  -3.0f),
        glm::vec3(1.0f,  -1.0f,  -3.0f),

        glm::vec3(-1.0f, -2.0f,  -3.0f),
        glm::vec3(0.0f, -2.0f,  -3.0f),
        glm::vec3(1.0f,  -2.0f,  -3.0f),

};



/// <summary>
/// Input process czyli wykonywanie się poszczególnych funkcji na wejœciu różnych przycisków.
/// </summary>
void processInput(GLFWwindow* window)
{
    if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS)
        glfwSetWindowShouldClose(window, true);
    if (glfwGetKey(window, GLFW_KEY_UP) == GLFW_PRESS)
        camera.ProcessKeyboard(FORWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_DOWN) == GLFW_PRESS)
        camera.ProcessKeyboard(BACKWARD, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_LEFT) == GLFW_PRESS)
        camera.ProcessKeyboard(LEFT, deltaTime);
    if (glfwGetKey(window, GLFW_KEY_RIGHT) == GLFW_PRESS)
        camera.ProcessKeyboard(RIGHT, deltaTime);
}


/// <summary>
/// Buffer ustawiania rozmiaru.
/// </summary>
void framebuffer_size_callback(GLFWwindow* window, int width, int height)
{
    glViewport(0, 0, width, height);
}

/// <summary>
/// Poruszanie myszką, dzięki temu zmienia widok.
/// </summary>
void mouse_callback(GLFWwindow* window, double xposIn, double yposIn)
{
    float xpos = static_cast<float>(xposIn);
    float ypos = static_cast<float>(yposIn);

    if (firstMouse)
    {
        lastX = xpos;
        lastY = ypos;
        firstMouse = false;
    }

    float xoffset = xpos - lastX;
    float yoffset = lastY - ypos;

    lastX = xpos;
    lastY = ypos;

    camera.ProcessMouseMovement(xoffset, yoffset);
}

/// <summary>
/// Poruszanie kamery na scrollu.
/// </summary>
void scroll_callback(GLFWwindow* window, double xoffset, double yoffset)
{
    camera.ProcessMouseScroll(static_cast<float>(yoffset));
}

/// <summary>
/// Załadowanie skyboxa czyli wyglądu który jest na całym programie.
/// </summary>
unsigned int loadCubemap()
{
    unsigned int textureID;
    glGenTextures(1, &textureID);
    glBindTexture(GL_TEXTURE_CUBE_MAP, textureID);
    int width, height, nrChannels;
    for (unsigned int i = 0; i < 6; i++)
    {
        unsigned char* data = stbi_load("skybox/right.jpg", &width, &height, &nrChannels, 0);
        if (i == 0)
            data = stbi_load("skybox/right.jpg", &width, &height, &nrChannels, 0);
        else if (i == 1)
            data = stbi_load("skybox/left.jpg", &width, &height, &nrChannels, 0);
        else if (i == 2)
            data = stbi_load("skybox/top.jpg", &width, &height, &nrChannels, 0);
        else if (i == 3)
            data = stbi_load("skybox/bottom.jpg", &width, &height, &nrChannels, 0);
        else if (i == 4)
            data = stbi_load("skybox/front.jpg", &width, &height, &nrChannels, 0);
        else if (i == 5)
            data = stbi_load("skybox/back.jpg", &width, &height, &nrChannels, 0);

        if (data)
        {
            glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
            stbi_image_free(data);
        }
        else
        {
            std::cout << "Cubemap texture failed to load at path: " << std::endl;
            stbi_image_free(data);
        }
    }
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_EDGE);

    return textureID;
}


/// <summary>
/// Nasłuchiwanie jeżeli przycisk został kliknięty to wykona się ta funkcja oraz zagnieżdzone funkcje w niej (tylko raz).
/// Key - przycisk
/// Action - akcja np. GLFW_PRESS
/// </summary>
void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    // Sterowanie kostką
    if (key == GLFW_KEY_1 && action == GLFW_PRESS)
        turn_cube_up_to_down(0, 3, true);
    if (key == GLFW_KEY_2 && action == GLFW_PRESS)
        turn_cube_up_to_down(1, 3, true);
    if (key == GLFW_KEY_3 && action == GLFW_PRESS)
        turn_cube_up_to_down(2, 3, true);
    if (key == GLFW_KEY_Q && action == GLFW_PRESS)
        turn_cube_down_to_up(0, 3, true);
    if (key == GLFW_KEY_W && action == GLFW_PRESS)
        turn_cube_down_to_up(1, 3, true);
    if (key == GLFW_KEY_E && action == GLFW_PRESS)
        turn_cube_down_to_up(2, 3, true);
    if (key == GLFW_KEY_A && action == GLFW_PRESS)
        turn_cube_left_to_right(0, 9, true);
    if (key == GLFW_KEY_S && action == GLFW_PRESS)
        turn_cube_left_to_right(3, 9, true);
    if (key == GLFW_KEY_D && action == GLFW_PRESS)
        turn_cube_left_to_right(6, 9, true);
    if (key == GLFW_KEY_Z && action == GLFW_PRESS)
        turn_cube_right_to_left(0, 9, true);
    if (key == GLFW_KEY_X && action == GLFW_PRESS)
        turn_cube_right_to_left(3, 9, true);
    if (key == GLFW_KEY_C && action == GLFW_PRESS)
        turn_cube_right_to_left(6, 9, true);

    // Pokazywanie podpowiedzi
    if ((key == GLFW_KEY_L && action == GLFW_PRESS) && isSolved == false)
        show_solve();

    // Mieszanie
    if ((glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS) && is_cube_solved() == true)
        mix_the_cube(1);
    if ((glfwGetKey(window, GLFW_KEY_O) == GLFW_PRESS) && is_cube_solved() == true)
        mix_the_cube(2);
    if ((glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS) && is_cube_solved() == true)
        mix_the_cube(3);

    // Prezentacja ułożenia kostki
    if (glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS) {
        cube_arranged(true);
        turn_cube_to_full();
    }

    // Zmiana koloru kostki
    if (glfwGetKey(window, GLFW_KEY_8) == GLFW_PRESS)
        color = 3;
    if (glfwGetKey(window, GLFW_KEY_9) == GLFW_PRESS)
        color = 2;
    if (glfwGetKey(window, GLFW_KEY_0) == GLFW_PRESS)
        color = 1;
}

/// <summary>
/// Ułożenie kostki do wersji domyślnej.
/// </summary>
void turn_cube_to_full()
{
    for (int i = 0; i < 27; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            sideCube[i][j] = j + 1;
        }
    }
}

/// <summary>
/// Obracanie kostki na przycisk od strony górnej do strony dolnej.
/// </summary>
void turn_cube_up_to_down(int which, int each, bool game) {
    if (game) count_moves += 1;
    int a, b, c;
    a = sideCube[which][5];
    b = sideCube[which + 9][5];
    c = sideCube[which + 2 * 9][5];
    sideCube[which][5] = sideCube[which + 2 * 9][0];
    sideCube[which + 9][5] = sideCube[which + 2 * 9 + each][0];
    sideCube[which + 2 * 9][5] = sideCube[which + 2 * 9 + 2 * each][0];

    sideCube[which + 2 * 9][0] = sideCube[which + 2 * each + 2 * 9][4];
    sideCube[which + 2 * 9 + each][0] = sideCube[which + 2 * each + 9][4];
    sideCube[which + 2 * 9 + 2 * each][0] = sideCube[which + 2 * each][4];

    sideCube[which + 2 * each][4] = sideCube[which][1];
    sideCube[which + 2 * each + 9][4] = sideCube[which + each][1];
    sideCube[which + 2 * each + 2 * 9][4] = sideCube[which + 2 * each][1];

    sideCube[which][1] = c;
    sideCube[which + each][1] = b;
    sideCube[which + 2 * each][1] = a;

    for (int i = 2; i < 4; i++)
    {
        a = sideCube[which][i];
        b = sideCube[which + 9][i];
        c = sideCube[which + 2 * 9][i];

        sideCube[which][i] = sideCube[which + 2 * 9][i];
        sideCube[which + 9][i] = sideCube[which + 2 * 9 + each][i];
        sideCube[which + 2 * 9][i] = sideCube[which + 2 * 9 + 2 * each][i];

        sideCube[which + 2 * 9][i] = sideCube[which + 2 * each + 2 * 9][i];
        sideCube[which + 2 * 9 + each][i] = sideCube[which + 2 * each + 9][i];
        sideCube[which + 2 * 9 + 2 * each][i] = sideCube[which + 2 * each][i];


        sideCube[which + 2 * each + 2 * 9][i] = sideCube[which + 2 * each][i];
        sideCube[which + 2 * each + 9][i] = sideCube[which + each][i];
        sideCube[which + 2 * each][i] = sideCube[which][i];

        sideCube[which][i] = c;
        sideCube[which + each][i] = b;
        sideCube[which + 2 * each][i] = a;
    }
    if (is_cube_solved() == true) cube_arranged(false);
}

/// <summary>
/// Obracanie kostki na przycisk od strony dolnej do strony górnej.
/// </summary>
void turn_cube_down_to_up(int which, int each, bool game) {
    if (game) count_moves += 1;
    int a, b, c;
    a = sideCube[which][5];
    b = sideCube[which + 9][5];
    c = sideCube[which + 2 * 9][5];

    sideCube[which][5] = sideCube[which + 2 * each][1];
    sideCube[which + 9][5] = sideCube[which + each][1];
    sideCube[which + 2 * 9][5] = sideCube[which][1];

    sideCube[which][1] = sideCube[which + 2 * each][4];
    sideCube[which + each][1] = sideCube[which + 2 * each + 9][4];
    sideCube[which + 2 * each][1] = sideCube[which + 2 * each + 2 * 9][4];

    sideCube[which + 2 * each][4] = sideCube[which + 2 * 9 + 2 * each][0];
    sideCube[which + 2 * each + 9][4] = sideCube[which + 2 * 9 + each][0];
    sideCube[which + 2 * each + 2 * 9][4] = sideCube[which + 2 * 9][0];

    sideCube[which + 2 * 9][0] = a;
    sideCube[which + 2 * 9 + each][0] = b;
    sideCube[which + 2 * 9 + 2 * each][0] = c;

    for (int i = 2; i < 4; i++)
    {
        a = sideCube[which][i];
        b = sideCube[which + 9][i];
        c = sideCube[which + 2 * 9][i];

        sideCube[which][i] = sideCube[which + 2 * each][i];
        sideCube[which + 9][i] = sideCube[which + each][i];
        sideCube[which + 2 * 9][i] = sideCube[which][i];

        sideCube[which][i] = sideCube[which + 2 * each][i];
        sideCube[which + each][i] = sideCube[which + 2 * each + 9][i];
        sideCube[which + 2 * each][i] = sideCube[which + 2 * each + 2 * 9][i];

        sideCube[which + 2 * each][i] = sideCube[which + 2 * 9 + 2 * each][i];
        sideCube[which + 2 * each + 9][i] = sideCube[which + each + 2 * 9][i];
        sideCube[which + 2 * each + 2 * 9][i] = sideCube[which + 2 * 9][i];

        sideCube[which + 2 * 9][i] = a;
        sideCube[which + each + 2 * 9][i] = b;
        sideCube[which + 2 * each + 2 * 9][i] = c;
    }
    if (is_cube_solved() == true) cube_arranged(false);
}

/// <summary>
/// Obracanie kostki na przycisk od strony lefej do strony prawej.
/// </summary>
void turn_cube_left_to_right(int which, int each, bool game) {
    if (game) count_moves += 1;
    int a, b, c;

    a = sideCube[which][2];
    b = sideCube[which + each][2];
    c = sideCube[which + 2 * each][2];

    sideCube[which][2] = sideCube[which + 2 * each][0];
    sideCube[which + each][2] = sideCube[which + 1 + 2 * each][0];
    sideCube[which + 2 * each][2] = sideCube[which + 2 + 2 * each][0];

    sideCube[which + 2 * each][0] = sideCube[which + 2 + 2 * each][3];
    sideCube[which + 1 + 2 * each][0] = sideCube[which + 2 + 1 * each][3];
    sideCube[which + 2 + 2 * each][0] = sideCube[which + 2][3];

    sideCube[which + 2 + 2 * each][3] = sideCube[which + 2][1];
    sideCube[which + 2 + 1 * each][3] = sideCube[which + 1][1];
    sideCube[which + 2][3] = sideCube[which][1];

    sideCube[which + 2][1] = a;
    sideCube[which + 1][1] = b;
    sideCube[which][1] = c;

    for (int i = 4; i < 6; i++)
    {
        a = sideCube[which][i];
        b = sideCube[which + each][i];
        c = sideCube[which + 2 * each][i];

        sideCube[which][i] = sideCube[which + 2 * each][i];
        sideCube[which + each][i] = sideCube[which + 1 + 2 * each][i];
        sideCube[which + 2 * each][i] = sideCube[which + 2 + 2 * each][i];

        sideCube[which + 2 * each][i] = sideCube[which + 2 + 2 * each][i];
        sideCube[which + 1 + 2 * each][i] = sideCube[which + 2 + 1 * each][i];
        sideCube[which + 2 + 2 * each][i] = sideCube[which + 2][i];

        sideCube[which + 2 + 2 * each][i] = sideCube[which + 2][i];
        sideCube[which + 2 + 1 * each][i] = sideCube[which + 1][i];
        sideCube[which + 2][i] = sideCube[which][i];

        sideCube[which + 2][i] = a;
        sideCube[which + 1][i] = b;
        sideCube[which][i] = c;
    }
    if (is_cube_solved() == true) cube_arranged(false);
}

/// <summary>
/// Obracanie kostki na przycisk od strony prawej do strony lewej.
/// </summary>
void turn_cube_right_to_left(int which, int each, bool game) {
    if (game) count_moves += 1;
    int a, b, c;

    a = sideCube[which][2];
    b = sideCube[which + each][2];
    c = sideCube[which + 2 * each][2];

    sideCube[which + 2 * each][2] = sideCube[which][1];
    sideCube[which + each][2] = sideCube[which + 1][1];
    sideCube[which][2] = sideCube[which + 2][1];

    sideCube[which][1] = sideCube[which + 2][3];
    sideCube[which + 1][1] = sideCube[which + 2 + each][3];
    sideCube[which + 2][1] = sideCube[which + 2 + 2 * each][3];

    sideCube[which + 2][3] = sideCube[which + 2 + 2 * each][0];
    sideCube[which + 2 + each][3] = sideCube[which + 1 + 2 * each][0];
    sideCube[which + 2 + 2 * each][3] = sideCube[which + 2 * each][0];

    sideCube[which + 2 + 2 * each][0] = c;
    sideCube[which + 1 + 2 * each][0] = b;
    sideCube[which + 2 * each][0] = a;

    for (int i = 4; i < 6; i++)
    {
        a = sideCube[which][i];
        b = sideCube[which + each][i];
        c = sideCube[which + 2 * each][i];

        sideCube[which + 2 * each][i] = sideCube[which][i];
        sideCube[which + each][i] = sideCube[which + 1][i];
        sideCube[which][i] = sideCube[which + 2][i];

        sideCube[which][i] = sideCube[which + 2][i];
        sideCube[which + 1][i] = sideCube[which + 2 + each][i];
        sideCube[which + 2][i] = sideCube[which + 2 + 2 * each][i];

        sideCube[which + 2][i] = sideCube[which + 2 + 2 * each][i];
        sideCube[which + 2 + each][i] = sideCube[which + 1 + 2 * each][i];
        sideCube[which + 2 + 2 * each][i] = sideCube[which + 2 * each][i];

        sideCube[which + 2 + 2 * each][i] = c;
        sideCube[which + 1 + 2 * each][i] = b;
        sideCube[which + 2 * each][i] = a;
    }
    if (is_cube_solved() == true) cube_arranged(false);
}

/// <summary>
/// Wypisanie kostki do konsoli  na podstawie cyfr.
/// </summary>
void print_cube_color() {
    for (int i = 0; i < 27; i++)
    {
        std::cout << i << ": " << sideCube[i][0] << sideCube[i][1] << sideCube[i][2] << sideCube[i][3] << sideCube[i][4] << sideCube[i][5] << std::endl;
    }
}

/// <summary>
/// Funkcja miesza ułożenie kostki którą następnie można układać
/// Są trzy tryby: easy, medium oraz hard czyli w zależności od wybrania preferencji, kostka bêdzie bardziej lub mnie trudna w ułożeniu.
/// </summary>
void mix_the_cube(int mode) {
    if (arranging == false) {
        isSolved = false;
        int number_of_changes = 0;

        // easy mode
        if (mode == 1) {
            number_of_changes = 5;
            random_moves = number_of_changes;
        }

        // medium mode
        if (mode == 2) {
            number_of_changes = 15;
            random_moves = number_of_changes;
        }

        // hard mode
        else if (mode == 3) {
            random_moves = (rand() % 36) + 15;
            number_of_changes = random_moves;
        }

        int random;
        for (int i = 0; i < number_of_changes; i++)
        {
            random = rand() % 12;

            switch (random)
            {
            case 0:
                solve += "Q ";
                turn_cube_up_to_down(0, 3, false);
                break;
            case 1:
                solve += "W ";
                turn_cube_up_to_down(1, 3, false);
                break;
            case 2:
                solve += "E ";
                turn_cube_up_to_down(2, 3, false);
                break;
            case 3:
                solve += "1 ";
                turn_cube_down_to_up(0, 3, false);
                break;
            case 4:
                solve += "2 ";
                turn_cube_down_to_up(1, 3, false);
                break;
            case 5:
                solve += "3 ";
                turn_cube_down_to_up(2, 3, false);
                break;
            case 6:
                solve += "Z ";
                turn_cube_left_to_right(0, 9, false);
                break;
            case 7:
                solve += "X ";
                turn_cube_left_to_right(3, 9, false);
                break;
            case 8:
                solve += "C ";
                turn_cube_left_to_right(6, 9, false);
                break;
            case 9:
                solve += "A ";
                turn_cube_right_to_left(0, 9, false);
                break;
            case 10:
                solve += "S ";
                turn_cube_right_to_left(3, 9, false);
                break;
            case 11:
                solve += "D ";
                turn_cube_right_to_left(6, 9, false);
                break;
            default:
                break;
            }
        }
        arranging = true;
        cout << "\n\nStart" << endl;
        start_timer = chrono::high_resolution_clock::now();
    }
}

/// <summary>
/// Sprawdzanie ułożenia kostki w przypdku ułożenia w lepszym czasie, wynik zapisywany zostaje w rankingu najlepszych czasów.
/// </summary>
void cube_arranged(bool skip) {
    if (arranging == true) {
        chrono::high_resolution_clock::time_point end_timer = std::chrono::high_resolution_clock::now();
        duration = end_timer - start_timer;
        if (skip == false) cout << "Czas trwania: " << duration.count() << " sekundy" << endl;
        else count_moves += random_moves;
        cout << "Ilosc ruchow: " << count_moves << endl << endl;

        arranging = false;
        if (duration.count() < top_scores[TOP_SCORES_COUNT - 1] && skip == false) {
            cout << "Gratulacje, udalo ci sie pobic rekord!!!\n" << endl;

            top_scores[TOP_SCORES_COUNT - 1] = duration.count();
            sort(top_scores, top_scores + TOP_SCORES_COUNT);

            ofstream file("best_score.txt");
            if (file.good()) {
                for (int i = 0; i < TOP_SCORES_COUNT; i++) {
                    file << fixed << setprecision(2) << top_scores[i] << endl;
                }
                file.close();
            }
            else {
                cout << "Nie mozna zapisac do pliku" << endl;
            }
        }
        show_best_scores();
        cout << endl << endl;
        show_options();
    }
    isSolved = true;
    random_moves = 0;
    count_moves = 0;
    solve = "";
}

/// <summary>
/// Wpisanie ustanowionego rekordu ułożenia kostki do rankingu najlepszych czasów.
/// </summary>
void set_best_scores() {
    ifstream file("best_score.txt");
    if (!file.good()) {
        cout << "\nNie mozna otworzyc pliku" << endl;
        file.close();
    }
    else {
        for (int i = 0; i < TOP_SCORES_COUNT; i++) {
            if (!(file >> top_scores[i])) break;
        }
        file.close();
        show_best_scores();
    }
}

/// <summary>
/// Wyświetlanie najlepszych czasów ułożenia kostki rubika.
/// </summary>
void show_best_scores() {
    std::cout << "Najlepsze wyniki:" << std::endl;
    for (int i = 0; i < TOP_SCORES_COUNT; i++) {
        int total_seconds = static_cast<int>(top_scores[i]);
        int minutes = total_seconds / 60;
        int seconds = total_seconds % 60;
        int milliseconds = static_cast<int>((top_scores[i] - total_seconds) * 1000);
        std::cout << "  " << i + 1 << ". " << minutes << ":" << std::setw(2) << std::setfill('0') << seconds << "." << std::setw(2) << std::setfill('0') << milliseconds / 10 << std::endl;
    }
}

/// <summary>
/// Wyświetlanie instrukcji obsługi programu w konsoli.
/// </summary>
void show_options() {
    cout << "Ruchy na kosce:" << endl;
    cout << "  1 - F    2 - S    3 - B'" << endl;
    cout << "  Q - F'   W - S'   E - B" << endl;
    cout << "  A - U'   S - E    D - D" << endl;
    cout << "  Z - U    X - E'   C - D'" << endl << endl;

    cout << "Opcje aplikacji:" << endl;
    cout << "  I     - pomieszanie kostki easy (5  iteracji)" << endl;
    cout << "  O     - pomieszanie kostki medium (15  iteracji)" << endl;
    cout << "  P     - pomieszanie kostki hard (15+ iteracji)" << endl;
    cout << "  L     - wyswietlenie podpowiedzi" << endl;
    cout << "  SPACE - symulacja ulozenie kostki" << endl;
    cout << "  0     - domyslny wyglad kostki" << endl;
    cout << "  9     - tryb dla daltonistow (deuteranopia)" << endl;
    cout << "  8     - tryb dla daltonistow (tritanopia)" << endl << endl;
}

/// <summary>
/// Funkcja sprawdza czy kostka rubika została ułożona.
/// </summary>
bool is_cube_solved()
{
    for (int i = 0; i < 27; i++)
    {
        for (int j = 0; j < 6; j++)
        {
            if (sideCube[i][j] != j + 1) return false;
        }
    }
    return true;
}

///<summary>
///Funkcja wyświetlająca podpowiedź do ułożenia kostki.
///</summary>

void show_solve() {
    isSolved = true;
    cout << "Podpowiedz: " << solve << "  <-----\n";
}

/// <summary>
/// Ustawienie parametrów dla tekstur.
/// </summary>
void dataTextureLoad() {
    // ustawianie parametrów wrap oraz repeat
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
    // ustawienie filtrowania oraz liniowej tekstury.
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    stbi_set_flip_vertically_on_load(true); // odwrócenie tekstury 
}

/// <summary>
/// LoadData generuje oraz zwraca teksture.
/// </summary>
void loadData(unsigned char* data, int width, int height) {
    if (data) {
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, width, height, 0, GL_RGB, GL_UNSIGNED_BYTE, data);
        glGenerateMipmap(GL_TEXTURE_2D);
    }
    else {
        std::cout << "Failed to load texture" << std::endl;
    }
    stbi_image_free(data);
}

/// <summary>
/// Funkcja ładująca textury które są nakładane na kostkę
/// </summary>
void loadTextures() {
    int width, height, nrChannels;

    // textureClassic1
    glGenTextures(1, &textureClassic1);
    glBindTexture(GL_TEXTURE_2D, textureClassic1);
    dataTextureLoad();
    unsigned char* dataClassic = stbi_load("colors/classic/red.png", &width, &height, &nrChannels, 0);
    loadData(dataClassic, width, height);


    // textureClassic2
    glGenTextures(1, &textureClassic2);
    glBindTexture(GL_TEXTURE_2D, textureClassic2);
    dataTextureLoad();
    dataClassic = stbi_load("colors/classic/green.png", &width, &height, &nrChannels, 0);
    loadData(dataClassic, width, height);


    // textureClassic3
    glGenTextures(1, &textureClassic3);
    glBindTexture(GL_TEXTURE_2D, textureClassic3);
    dataTextureLoad();
    dataClassic = stbi_load("colors/classic/blue.png", &width, &height, &nrChannels, 0);
    loadData(dataClassic, width, height);


    // textureClassic4
    glGenTextures(1, &textureClassic4);
    glBindTexture(GL_TEXTURE_2D, textureClassic4);
    dataTextureLoad();
    dataClassic = stbi_load("colors/classic/yellow.png", &width, &height, &nrChannels, 0);
    loadData(dataClassic, width, height);


    // textureClassic5
    glGenTextures(1, &textureClassic5);
    glBindTexture(GL_TEXTURE_2D, textureClassic5);
    dataTextureLoad();
    dataClassic = stbi_load("colors/classic/orange.png", &width, &height, &nrChannels, 0);
    loadData(dataClassic, width, height);


    // textureClassic6 
    glGenTextures(1, &textureClassic6);
    glBindTexture(GL_TEXTURE_2D, textureClassic6);
    dataTextureLoad();
    dataClassic = stbi_load("colors/classic/white.png", &width, &height, &nrChannels, 0);
    loadData(dataClassic, width, height);



    // textureDeuteranopia1
    glGenTextures(1, &textureDeuteranopia1);
    glBindTexture(GL_TEXTURE_2D, textureDeuteranopia1);
    dataTextureLoad();
    unsigned char* dataDeuteranopia = stbi_load("colors/deuteranopia/red.png", &width, &height, &nrChannels, 0);
    loadData(dataDeuteranopia, width, height);


    // textureDeuteranopia2
    glGenTextures(1, &textureDeuteranopia2);
    glBindTexture(GL_TEXTURE_2D, textureDeuteranopia2);
    dataTextureLoad();
    dataDeuteranopia = stbi_load("colors/deuteranopia/green.png", &width, &height, &nrChannels, 0);
    loadData(dataDeuteranopia, width, height);


    // textureDeuteranopia3
    glGenTextures(1, &textureDeuteranopia3);
    glBindTexture(GL_TEXTURE_2D, textureDeuteranopia3);
    dataTextureLoad();
    dataDeuteranopia = stbi_load("colors/deuteranopia/blue.png", &width, &height, &nrChannels, 0);
    loadData(dataDeuteranopia, width, height);


    // textureDeuteranopia4
    glGenTextures(1, &textureDeuteranopia4);
    glBindTexture(GL_TEXTURE_2D, textureDeuteranopia4);
    dataTextureLoad();
    dataDeuteranopia = stbi_load("colors/deuteranopia/yellow.png", &width, &height, &nrChannels, 0);
    loadData(dataDeuteranopia, width, height);


    // textureDeuteranopia5
    glGenTextures(1, &textureDeuteranopia5);
    glBindTexture(GL_TEXTURE_2D, textureDeuteranopia5);
    dataTextureLoad();
    dataDeuteranopia = stbi_load("colors/deuteranopia/orange.png", &width, &height, &nrChannels, 0);
    loadData(dataDeuteranopia, width, height);


    // textureDeuteranopia6
    glGenTextures(1, &textureDeuteranopia6);
    glBindTexture(GL_TEXTURE_2D, textureDeuteranopia6);
    dataTextureLoad();
    dataDeuteranopia = stbi_load("colors/deuteranopia/white.png", &width, &height, &nrChannels, 0);
    loadData(dataDeuteranopia, width, height);


    // textureTritanopia1
    glGenTextures(1, &textureTritanopia1);
    glBindTexture(GL_TEXTURE_2D, textureTritanopia1);
    dataTextureLoad();
    unsigned char* dataTritanopia = stbi_load("colors/tritanopia/red.png", &width, &height, &nrChannels, 0);
    loadData(dataTritanopia, width, height);


    // textureTritanopia2
    glGenTextures(1, &textureTritanopia2);
    glBindTexture(GL_TEXTURE_2D, textureTritanopia2);
    dataTextureLoad();
    dataTritanopia = stbi_load("colors/tritanopia/green.png", &width, &height, &nrChannels, 0);
    loadData(dataTritanopia, width, height);


    // textureTritanopia3
    glGenTextures(1, &textureTritanopia3);
    glBindTexture(GL_TEXTURE_2D, textureTritanopia3);
    dataTextureLoad();
    dataTritanopia = stbi_load("colors/tritanopia/blue.png", &width, &height, &nrChannels, 0);
    loadData(dataTritanopia, width, height);


    // textureTritanopia4
    glGenTextures(1, &textureTritanopia4);
    glBindTexture(GL_TEXTURE_2D, textureTritanopia4);
    dataTextureLoad();
    dataTritanopia = stbi_load("colors/tritanopia/yellow.png", &width, &height, &nrChannels, 0);
    loadData(dataTritanopia, width, height);


    // textureTritanopia5
    glGenTextures(1, &textureTritanopia5);
    glBindTexture(GL_TEXTURE_2D, textureTritanopia5);
    dataTextureLoad();
    dataTritanopia = stbi_load("colors/tritanopia/orange.png", &width, &height, &nrChannels, 0);
    loadData(dataTritanopia, width, height);


    // textureTritanopia6
    glGenTextures(1, &textureTritanopia6);
    glBindTexture(GL_TEXTURE_2D, textureTritanopia6);
    dataTextureLoad();
    dataTritanopia = stbi_load("colors/tritanopia/white.png", &width, &height, &nrChannels, 0);
    loadData(dataTritanopia, width, height);

}