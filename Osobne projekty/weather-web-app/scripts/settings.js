const settingsButton = document.querySelector('.container_settings');
const modal = document.getElementById('modal');

settingsButton.addEventListener('click', function() {
    modal.style.display = 'block';
});

const closeBtn = document.querySelector('.close');
closeBtn.addEventListener('click', function() {
    modal.style.display = 'none';

});

window.addEventListener('click', function(event) {
    if (event.target === modal) {
        modal.style.display = 'none';
    }
});