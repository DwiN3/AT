# UltimateScreenSeats

## Django API Setup

This project contains a Django application running inside a Docker container. Below are the steps to build and run the Docker image.

### Prerequisites
- [Docker](https://www.docker.com/) installed on your system
- A `Dockerfile` in the project directory
- A `requirements.txt` file with the application's dependencies

### Instructions

#### 1. Clone the Repository
Clone the repository to your local machine:

`git clone https://github.com/MichalPolak01/UltimateScreenSeats.git`
`cd ultimate-screen-seats-backend`

#### 2. Build the Docker Image

Run the following command to build the Docker image:

`docker build -t image-name .` 

-   `image-name` – the name you want to assign to the image.

#### 3. Run the Container

Run a container based on the built image:

`docker run -p 8000:8000 image-name` 

-   `-p 8000:8000` – maps port 8000 in the container to port 8000 on the host.
-   `image-name` – the name of the image created in the previous step.

#### 4. Access the Application

Once the container is running, the Django application will be available at:

`http://localhost:8000` 

#### 5. Update the Docker Image

If you make changes to your code and need to update the Docker image:

1.  Stop any running containers:
       
    `docker ps`
    `docker stop CONTAINER_ID` 
    
2.  Rebuild the Docker image:
       
    `docker build -t image-name .` 
    
3.  Start a new container with the updated image:
        
    `docker run -p 8000:8000 image-name` 
    

#### 6. Stop the Container

To stop the container, find its **CONTAINER ID**:

`docker ps` 

Then stop it:

`docker stop CONTAINER_ID`