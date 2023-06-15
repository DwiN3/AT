#include "cube.h"


/// <summary>
/// Główna funkcja aplikacji.
/// Zawiera ona inicjalizacje okna, VAO, VBO, przydzielenie shaderów.
/// Nakłada textury na poszczególne boki.
/// </summary>
int main()
{
    // pokazuje opcje
    show_options();
    // ustawia najlepsze czasy
    set_best_scores();

    // glfw
    glfwInit();
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

    // glfw window
    GLFWwindow* window = glfwCreateWindow(SCR_WIDTH, SCR_HEIGHT, "RubikCube", NULL, NULL);
    if (window == NULL)
    {
        std::cout << "Failed to create GLFW window" << std::endl;
        glfwTerminate();
        return -1;
    }
    glfwMakeContextCurrent(window);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
    glfwSetCursorPosCallback(window, mouse_callback);
    glfwSetScrollCallback(window, scroll_callback);

    // glfw przechwytywanie myszki
    glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);

    // glad ładowanie funkcji
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress))
    {
        std::cout << "Failed to initialize GLAD" << std::endl;
        return -1;
    }

    // konfiguracja globalnego stanu opengl
    glEnable(GL_DEPTH_TEST);

    // ładowanie shaderów
    Shader ourShader("assets/camera.vs", "assets/camera.fs");
    Shader skyboxShader("assets/skybox.vs", "assets/skybox.fs");

    // ustawianie pozycji
    float vertices[] = {
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 0.0f,

        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 1.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,

        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,

        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,
         0.5f, -0.5f, -0.5f,  1.0f, 1.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
         0.5f, -0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f, -0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f, -0.5f, -0.5f,  0.0f, 1.0f,

        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
         0.5f,  0.5f, -0.5f,  1.0f, 1.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
         0.5f,  0.5f,  0.5f,  1.0f, 0.0f,
        -0.5f,  0.5f,  0.5f,  0.0f, 0.0f,
        -0.5f,  0.5f, -0.5f,  0.0f, 1.0f,
    };


    // skybox
    float skyboxVertices[] = {         
        -1.0f,  1.0f, -1.0f,
        -1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f, -1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,

        -1.0f, -1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f, -1.0f,  1.0f,
        -1.0f, -1.0f,  1.0f,

        -1.0f,  1.0f, -1.0f,
         1.0f,  1.0f, -1.0f,
         1.0f,  1.0f,  1.0f,
         1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f,  1.0f,
        -1.0f,  1.0f, -1.0f,

        -1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f, -1.0f,
         1.0f, -1.0f, -1.0f,
        -1.0f, -1.0f,  1.0f,
         1.0f, -1.0f,  1.0f
    };
    // skybox VAO
    unsigned int skyboxVAO, skyboxVBO;
    glGenVertexArrays(1, &skyboxVAO);
    glGenBuffers(1, &skyboxVBO);
    glBindVertexArray(skyboxVAO);
    glBindBuffer(GL_ARRAY_BUFFER, skyboxVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(skyboxVertices), &skyboxVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);

    unsigned int cubemapTexture = loadCubemap();
    skyboxShader.use();
    skyboxShader.setInt("skybox", 0);

    unsigned int VBO, VAO;
    glGenVertexArrays(1, &VAO);
    glGenBuffers(1, &VBO);

    glBindVertexArray(VAO);

    glBindBuffer(GL_ARRAY_BUFFER, VBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

    // pozycja atrybutów
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    // kordy atrybutów tekstur
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), (void*)(3 * sizeof(float)));
    glEnableVertexAttribArray(1); 


    loadTextures();


    // główna pętla renderowania
    while (!glfwWindowShouldClose(window))
    {
        // czas na klatkę
        float currentFrame = static_cast<float>(glfwGetTime());
        deltaTime = currentFrame - lastFrame;
        lastFrame = currentFrame;

        // podpięcie inputów
        processInput(window);

        // aktywacja shaderów
        ourShader.use();

        glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


        // przekazywanie macierzy do shadera
        glm::mat4 projection = glm::perspective(glm::radians(camera.Zoom), (float)SCR_WIDTH / (float)SCR_HEIGHT, 0.1f, 100.0f);
        ourShader.setMat4("projection", projection);

        // kamera
        glm::mat4 view = camera.GetViewMatrix();
        ourShader.setMat4("view", view);

        int count = 6;
        // renderowanie 27 boxów
        glBindVertexArray(VAO);
        for (unsigned int i = 0; i < 27; i++)
        {
            ourShader.use();
            for (unsigned int j = 0; j < 6; j++)
            {
                // 0 - white
                // 1 - red
                // 2 - green
                // 3 - blue
                // 4 - yellow
                // 5 - orange
                // aktywacja tekstur dla poszczególnych boków
                if (sideCube[i][j] == 1) {
                    glActiveTexture(GL_TEXTURE1);
                    if (color == 1)
                        glBindTexture(GL_TEXTURE_2D, textureClassic1);
                    else if (color == 2)
                        glBindTexture(GL_TEXTURE_2D, textureDeuteranopia1);
                    else if (color == 3)
                        glBindTexture(GL_TEXTURE_2D, textureTritanopia1);
                    ourShader.setInt("texture1", 5);

                }
                else if (sideCube[i][j] == 2) {
                    glActiveTexture(GL_TEXTURE2);
                    if (color == 1)
                        glBindTexture(GL_TEXTURE_2D, textureClassic2);
                    else if (color == 2)
                        glBindTexture(GL_TEXTURE_2D, textureDeuteranopia2);
                    else if (color == 3) {
                        glBindTexture(GL_TEXTURE_2D, textureTritanopia2);
                    }
                    ourShader.setInt("texture1", 1);

                }
                else if (sideCube[i][j] == 3) {
                    glActiveTexture(GL_TEXTURE3);
                    if (color == 1)
                        glBindTexture(GL_TEXTURE_2D, textureClassic3);
                    else if (color == 2)
                        glBindTexture(GL_TEXTURE_2D, textureDeuteranopia3);
                    else if (color == 3)
                        glBindTexture(GL_TEXTURE_2D, textureTritanopia3);
                    ourShader.setInt("texture1", 2);

                }
                else if (sideCube[i][j] == 4) {
                    glActiveTexture(GL_TEXTURE4);
                    if (color == 1)
                        glBindTexture(GL_TEXTURE_2D, textureClassic4);
                    else if (color == 2)
                        glBindTexture(GL_TEXTURE_2D, textureDeuteranopia4);
                    else if (color == 3)
                        glBindTexture(GL_TEXTURE_2D, textureTritanopia4);
                    ourShader.setInt("texture1", 3);

                }

                else if (sideCube[i][j] == 5) {
                    glActiveTexture(GL_TEXTURE5);
                    if (color == 1)
                        glBindTexture(GL_TEXTURE_2D, textureClassic5);
                    else if (color == 2)
                        glBindTexture(GL_TEXTURE_2D, textureDeuteranopia5);
                    else if (color == 3)
                        glBindTexture(GL_TEXTURE_2D, textureTritanopia5);
                    ourShader.setInt("texture1", 4);

                }
                else if (sideCube[i][j] == 6) {
                    glActiveTexture(GL_TEXTURE6);
                    if (color == 1)
                        glBindTexture(GL_TEXTURE_2D, textureClassic6);
                    else if (color == 2)
                        glBindTexture(GL_TEXTURE_2D, textureDeuteranopia6);
                    else if (color == 3)
                        glBindTexture(GL_TEXTURE_2D, textureTritanopia6);
                    ourShader.setInt("texture1", 0);

                }

                // obliczanie modelu macierzy dla każdego obiektu i przekazanie go do shadera przed wyrysowaniem
                glm::mat4 model = glm::mat4(1.0f); 
                model = glm::translate(model, cubePositions[i]);
                model = glm::rotate(model, glm::radians(0.0f), glm::vec3(1.0f, 0.0f, 0.0f));
                ourShader.setMat4("model", model);
                // liczenie ilości wyrysowanych trójkątów i zmiana tekstury co każde 2 trójkąty
                glDrawArrays(GL_TRIANGLES, 0, count);
                count += 6;
            }
            // resetowanie liczenia trójkątów po wyrysowaniu całej małej kostki
            if (count >= 35)
                count = 6;
        }

        // rysowanie skyboxa
        glDepthFunc(GL_LEQUAL); 
        skyboxShader.use();
        view = glm::mat4(glm::mat3(camera.GetViewMatrix())); 
        skyboxShader.setMat4("view", view);
        skyboxShader.setMat4("projection", projection);
        // skybox cube
        glBindVertexArray(skyboxVAO);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_CUBE_MAP, cubemapTexture);
        glDrawArrays(GL_TRIANGLES, 0, 36);
        glBindVertexArray(0);
        glDepthFunc(GL_LESS); 

        // podpięcie inputów
        glfwSetKeyCallback(window, key_callback);
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
    // usunięcie zaalokowanych zasobów
    glDeleteVertexArrays(1, &VAO);
    glDeleteBuffers(1, &VBO);

    glDeleteVertexArrays(1, &skyboxVAO);
    glDeleteBuffers(1, &skyboxVBO);

    // glfw: czyszczenie poprzednich zasobów
    glfwTerminate();
    return 0;
}


