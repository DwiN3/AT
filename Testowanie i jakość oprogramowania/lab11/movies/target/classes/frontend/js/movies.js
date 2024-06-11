function isImage(url) {
    return url.endsWith(".jpg") || url.endsWith(".png");
}

function getPlaceholder() {
    return '../img/placeholder.png';
}

async function fetchMovies() {
    try {
        const response = await fetch('/api/movies');
        if (!response.ok) {
            throw new Error('Network response was not ok');
        }
        const data = await response.json();
        renderMovies(data);
    } catch (error) {
        document.getElementById('movies').innerHTML = "<div class='error-message'>Ups! Aplikacja nie odpowiada... :/</div>";
        console.error('Fetch error:', error);
    }
}

function renderMovies(data) {
    const html = data.map(movie => `
        <div class="card">
            <img src="${isImage(movie.image) ? movie.image : getPlaceholder()}" alt="movie image">
            <h3>${movie.title}</h3>
            <p class="title">Rok produkcji</p>
            <p>${movie.releaseYear !== undefined ? movie.releaseYear : '????'}</p>
            <p><button>ID: ${movie.movieId}</button></p>
        </div>
    `).join('');
    document.getElementById('movies').innerHTML = html;
}

fetchMovies();
