window.onload = function() {
    currentLang = 'pl';
    changeLanguage('pl');
    changeTemperature('C');
    currentUnit = "C";
    console.log(getLastSelectedTempUnit())
    changeLanguage(getLastSelectedLanguage())
    setDarkMode(getDarkModeFromLocalStorage());
    changeTemperature(getLastSelectedTempUnit())
    currentUnit = getLastSelectedTempUnit()
    handleUnitChange()


    display_mode('day');
}