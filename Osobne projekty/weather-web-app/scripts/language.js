function changeLanguage(lang) {
    const langTogglePL = document.getElementById('langTogglePL');
    const langToggleEN = document.getElementById('langToggleEN');
    const queryInput = document.getElementById('query');
    const dayButton = document.getElementById('day_button');
    const weekButton = document.getElementById('week_button');
    const tempText = document.getElementById('temperatureText');
    const lastSearches = document.getElementById('last_searches');

    if (lang === 'pl') {
        lastSearches.textContent = "Ostatnio wyszukiwane:"
        langTogglePL.checked = true;
        langToggleEN.checked = false;
        languageText.textContent = 'Język:';
        settingsText.textContent = 'Ustawienia';
        darkModeText.textContent = 'Tryb ciemny';
        queryInput.placeholder = 'Szukaj...';
        dayButton.textContent = 'Dzień';
        weekButton.textContent = 'Tydzień';
        tempText.textContent = "Temperatura";
       
    } else if (lang === 'en') {
        lastSearches.textContent = 'Recent searches:'
        langToggleEN.checked = true;
        langTogglePL.checked = false;
        languageText.textContent = 'Language:';
        settingsText.textContent = 'Settings';
        darkModeText.textContent = 'Dark mode';
        queryInput.placeholder = 'Search...';
        dayButton.textContent = 'Day';
        weekButton.textContent = 'Week';
    }
    
}

function toggleLanguage(selectedCheckbox) {
    const checkboxes = document.getElementsByName('langToggle');
    for (let checkbox of checkboxes) {
        if (checkbox !== selectedCheckbox) {
            checkbox.checked = false;
        }
    }

    if (!selectedCheckbox.checked) {
        selectedCheckbox.checked = true;
    }

    const selectedLang = selectedCheckbox.value;
    changeLanguage(selectedLang);
    setDaysLang(selectedLang);

    setLastSelectedLanguage(selectedLang);
}

function setLastSelectedLanguage(lang) {
    localStorage.setItem('lastSelectedLang', lang);
}

function getLastSelectedLanguage() {
    return localStorage.getItem('lastSelectedLang');
}

window.onload = function() {
    const langTogglePL = document.querySelector('input[value="pl"]');
    langTogglePL.checked = true;
    changeLanguage('pl');
};
function setDaysLang(lang) {
    if (lang !== currentLang) {
        currentLang = lang;
        handleLangChange();
    }
  }

