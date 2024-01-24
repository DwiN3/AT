// const content = document.getElementById("day_content")

// export function WeatherForDay(data, unit, type){
//     content.innerHTML = "";
//     let day = 0;
//     let numCards = 0;
//     if (type === "day") {
//         numCards = 24;
//     } else {
//         numCards = 7;
//     }
//     for (let i = 0; i < numCards; i++) {
//         let card = document.createElement("div");
//         card.classList.add("card");
//         let dayName = getHour(data[day].datetime);
//         if (type === "week") {
//           dayName = getDayName(data[day].datetime);
//         }
//         let dayTemp = data[day].temp;
//         if (unit === "f") {
//           dayTemp = celciusToFahrenheit(data[day].temp);
//         }
//         let iconCondition = data[day].icon;
//         let iconSrc = getIcon(iconCondition);
//         card.innerHTML = `
//             <div class="day_block">
//             <span class="day_text_block">${dayName}</span>
//             <img class="day_icon_block" src="${iconSrc}">
//             <span class="day_temp_block">${dayTemp}</span>
//             </div>
//         `;
//        content.appendChild(card);
//         day++;
//     }
// }
