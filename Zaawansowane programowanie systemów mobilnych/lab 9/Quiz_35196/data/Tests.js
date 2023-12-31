// Tests.js

class Tests {
  constructor(title, tags, description, tasks) {
    this.titleTest = title;
    this.tags = tags;
    this.description = description;
    this.tasks = tasks;
  }
}

const tasks = [
  {
    "question": "Który wódz po śmierci Gajusza Mariusza, prowadził wojnę domową z Sullą?",
    "answers": [
      {
        "content": "LUCJUSZ CYNNA",
        "isCorrect": true
      },
      {
        "content": "JULIUSZ CEZAR",
        "isCorrect": false
      },
      {
        "content": "LUCJUSZ MURENA",
        "isCorrect": false
      },
      {
        "content": "MAREK KRASSUS",
        "isCorrect": false
      }
    ],
    "duration": 30
  },
  {
    "question": "Kto był pierwszym królem Polski?",
    "answers": [
      {
        "content": "BOLESLAW CHROBRY",
        "isCorrect": false
      },
      {
        "content": "MIESZKO I",
        "isCorrect": true
      },
      {
        "content": "KAZIMIERZ WIELKI",
        "isCorrect": false
      },
      {
        "content": "JAN III SOBIESKI",
        "isCorrect": false
      }
    ],
    "duration": 45
  },
  {
    "question": "Które miasto jest stolicą Japonii?",
    "answers": [
      {
        "content": "SEOUL",
        "isCorrect": false
      },
      {
        "content": "BEIJING",
        "isCorrect": false
      },
      {
        "content": "TOKIO",
        "isCorrect": true
      },
      {
        "content": "HANOI",
        "isCorrect": false
      }
    ],
    "duration": 20
  },
  {
    "question": "Który pierwiastek chemiczny ma symbol 'O'?",
    "answers": [
      {
        "content": "TLLEN",
        "isCorrect": false
      },
      {
        "content": "OXYGEN",
        "isCorrect": true
      },
      {
        "content": "GOLD",
        "isCorrect": false
      },
      {
        "content": "IRON",
        "isCorrect": false
      }
    ],
    "duration": 25
  }
];

const animalTest = [
  {
    "question": "Ile nóg ma pająk?",
    "answers": [
      {
        "content": "Cztery",
        "isCorrect": false
      },
      {
        "content": "Sześć",
        "isCorrect": true
      },
      {
        "content": "Osiem",
        "isCorrect": false
      },
      {
        "content": "Dziesięć",
        "isCorrect": false
      }
    ],
    "duration": 20
  },
  {
    "question": "Które zwierzę jest największe na świecie?",
    "answers": [
      {
        "content": "Słoń afrykański",
        "isCorrect": false
      },
      {
        "content": "Błękitny wieloryb",
        "isCorrect": true
      },
      {
        "content": "Girafa",
        "isCorrect": false
      },
      {
        "content": "Hipopotam",
        "isCorrect": false
      }
    ],
    "duration": 30
  },
  {
    "question": "Jaki jest największy gatunek wśród kotowatych?",
    "answers": [
      {
        "content": "Lew",
        "isCorrect": false
      },
      {
        "content": "Tygrys syberyjski",
        "isCorrect": true
      },
      {
        "content": "Puma",
        "isCorrect": false
      },
      {
        "content": "Pantera",
        "isCorrect": false
      }
    ],
    "duration": 25
  },
  {
    "question": "Które zwierzę jest symbolem Australii?",
    "answers": [
      {
        "content": "Kangur",
        "isCorrect": true
      },
      {
        "content": "Koala",
        "isCorrect": false
      },
      {
        "content": "Emu",
        "isCorrect": false
      },
      {
        "content": "Wombat",
        "isCorrect": false
      }
    ],
    "duration": 20, 
  },
  {
    "question": "Ile kości ma dorosły człowiek?",
    "answers": [
      {
        "content": "206",
        "isCorrect": true
      },
      {
        "content": "180",
        "isCorrect": false
      },
      {
        "content": "250",
        "isCorrect": false
      },
      {
        "content": "300",
        "isCorrect": false
      }
    ],
    "duration": 25
  }
];

const languageTest = [
  {
    "question": "Które z poniższych języków należy do grupy języków romańskich?",
    "answers": [
      {
        "content": "Rosyjski",
        "isCorrect": false
      },
      {
        "content": "Hiszpański",
        "isCorrect": true
      },
      {
        "content": "Arabski",
        "isCorrect": false
      },
      {
        "content": "Chiński",
        "isCorrect": false
      }
    ],
    "duration": 30
  },
  {
    "question": "Który alfabet jest używany w języku greckim?",
    "answers": [
      {
        "content": "Łaciński",
        "isCorrect": false
      },
      {
        "content": "Cyrylica",
        "isCorrect": false
      },
      {
        "content": "Grecki alfabet",
        "isCorrect": true
      },
      {
        "content": "Arabski",
        "isCorrect": false
      }
    ],
    "duration": 25
  },
  {
    "question": "Który język ma popularne przekleństwo na literę K?",
    "answers": [
      {
        "content": "Francuski",
        "isCorrect": false
      },
      {
        "content": "Polski",
        "isCorrect": true
      },
      {
        "content": "Hiszpański",
        "isCorrect": false
      },
      {
        "content": "Niemiecki",
        "isCorrect": false
      }
    ],
    "duration": 10
  }
];



const TestsList = [
  new Tests("Show test", ['#Tag1', '#Tag2'], 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...', tasks),
  new Tests("Animal Quiz", ['#Tag3', '#Tag4'], 'Test wiedzy o zwierzętach\nWitaj w teście wiedzy o zwierzętach! Sprawdź, ile wiesz na temat różnorodnego świata fauny. Odpowiedz na pytania dotyczące różnych gatunków zwierząt, ich cech charakterystycznych i ciekawostek związanych z ich życiem. Czy uda Ci się osiągnąć doskonały wynik?', animalTest),
  new Tests("Language Quiz", ['#Tag5', '#Tag6'], 'Test Wiedzy Językowej\nRozwiń swoje umiejętności językowe poprzez udział w naszym fascynującym teście wiedzy językowej! Ta kategoria skupia się na różnorodnych aspektach języków, od ich pochodzenia po ciekawostki związane z konkretnymi słowami czy zwrotami. Sprawdź swoją erudycję lingwistyczną, odpowiadając na pytania dotyczące języków z różnych regionów i rodzin językowych. Czy potrafisz rozróżnić języki romańskie od innych grup językowych?', languageTest),
  new Tests(4, ['#Tag7', '#Tag8'], 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...', tasks),
  new Tests(5, ['#Tag7', '#Tag8'], 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...', tasks),
  new Tests(6, ['#Tag7', '#Tag8'], 'Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat...', tasks),
];

export { Tests, TestsList };
