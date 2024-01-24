const dayContent = document.getElementById('day_content');
const weekContent = document.getElementById('week_content');
const infoContent = document.getElementById('day_info_content');
const gridContent = document.getElementById('weather_main_content');
const dayButton = document.getElementById('day_button');
const weekButton = document.getElementById('week_button');
let show_info = false;
let visibilityInfo = false;
let mode="week";
console.log(mode);
  function getMode(){
    return mode;
  }
  function getVisibileInfo(){
    return show_info;
  }


  function display_mode(selected_mode) {
    if (selected_mode === 'day') {
      mode = "day"
      dayButton.style.color = '#aaccec';
      weekButton.style.color = '#898989';
      infoContent.style.display = 'none';
      dayContent.style.display = 'grid';
      weekContent.style.display = 'none';
      gridContent.style.gridTemplateColumns = 'repeat(6, 1fr)';
    } else if (selected_mode === 'week') {
      mode = "week"
      weekButton.style.color = '#aaccec';
      dayButton.style.color = '#898989';
      infoContent.style.display = 'none';
      weekContent.style.display = 'flex';
      dayContent.style.display = 'none';
    }
    show_info = false;
  }

  function show_today_info() {
    console.log(show_info)
    if (!show_info) {
      show_info = true;
      infoContent.style.display = 'flex';
      dayContent.style.display = 'none';
      weekContent.style.display = 'none';
      gridContent.style.gridTemplateColumns = 'repeat(3, 1fr)';
      Array.from(dayContent.getElementsByClassName('day_block')).forEach((block) => {
        block.classList.add('flip-in');
      });
      Array.from(weekContent.getElementsByClassName('week_block')).forEach((block) => {
        block.classList.add('flip-in');
      });
      Array.from(infoContent.getElementsByClassName('day_info_block')).forEach((block) => {
        block.classList.add('flip-in');
      });

    } else {

      show_info = false;
      infoContent.style.display = 'none';
      if (mode === 'day') {
        display_mode('day');
      } else {
        display_mode('week');
      }
  
      Array.from(dayContent.getElementsByClassName('day_block')).forEach((block) => {
        block.classList.add('flip-in');
      });
      Array.from(weekContent.getElementsByClassName('week_block')).forEach((block) => {
        block.classList.add('flip-in');
      });
      Array.from(infoContent.getElementsByClassName('day_info_block')).forEach((block) => {
        block.classList.add('flip-out');
        setTimeout(() => {
          block.classList.remove('flip-in');
          block.classList.remove('flip-out');
        }, 500);
      });
    }

  }
  
  display_mode('day');
  infoContent.style.display = 'none';