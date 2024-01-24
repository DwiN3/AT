const time = document.getElementById("info_day")

function getDayName(date, lang) {
  let day = new Date(date);
  let days = {
    pl: ["Niedziela", "Poniedziałek", "Wtorek", "Środa", "Czwartek", "Piątek", "Sobota"],
    en: ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
  };
  return days[lang][day.getDay()];
}

function getInfo(nr, lang) {
  let info = {
    pl: ["Indeks UV", "Prędkość wiatru", "Wschód / Zachód", "Wilgotność", "Widoczność", "Jakość Powietrza"],
    en: ["Index UV", "Wind speed", "Sunrise / Sunset", "Humidity", "Visibility", "Air Quality"],
  };
  return info[lang][nr]
}

function getDateTime() {
  let now = new Date(),
    hour = now.getHours(),
    minute = now.getMinutes();

  // 12 hours format
  hour = hour % 12;
  if (hour < 10) {
    hour = "0" + hour;
  }
  if (minute < 10) {
    minute = "0" + minute;
  }
  let dayString = getDayName(now, currentLang);
  return `${dayString}, ${hour}:${minute}`;
}

time.innerText = getDateTime();
setInterval(() => {
  time.innerText = getDateTime();
}, 1000);