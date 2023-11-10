const math = require('mathjs');

//  AC
export const clearDisplay = (setDisplayText) => {
  setDisplayText('0');
};


//  =
export const calculateResult = (displayText,setDisplayText) => {
  try {
    const text = displayText;
    let result;

    if (text.includes("^")) {
      result = eval(text.replace(/\^/g, "**"));
    } else if (text.includes("√")) {
      const parts = text.split("√");
      if (parts.length === 2) {
        const degree = parseFloat(parts[0]);
        const operand = parseFloat(parts[1]);
        if (!isNaN(degree) && !isNaN(operand)) {
          result = Math.pow(operand, 1 / degree);
        }
      }
    } else {
      result = math.evaluate(text);
    }
    
    if (!isNaN(result)) {
      setDisplayText(result.toString());
    } else {
      setDisplayText('Błąd obliczeń');
    }
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  ^
export const power = (displayText,setDisplayText, exponent) => {
    let result;
      const base = parseFloat(displayText);
      const exp = parseFloat(exponent);
      if (!isNaN(base) && !isNaN(exp)) {
        result = Math.pow(base, exp);
        if (!isNaN(result)) {
          setDisplayText(result.toString());
        } else {
          setDisplayText('Błąd obliczeń');
        }
      } else {
        setDisplayText('Błąd obliczeń');
      }
};


//  sqrt
export const sqrt = (displayText, setDisplayText, exponent) => {
  let base = '';
  let index = '';
  if (exponent.includes('√')) {
    base = displayText;
    index = 2;
  } else if (exponent.includes('³√')) {
    base = displayText;
    index = 3; 
  }

  if (base !== '' && !isNaN(index)) {
    const result = Math.pow(parseFloat(base), 1 / index);
    setDisplayText(result.toString());
  } else {
    setDisplayText('Invalid input');
  }
};


//  %
export const procent = (displayText,setDisplayText) => {
  try {
    const currentText = eval(displayText);
    const finalResult = currentText / 100;
    setDisplayText(finalResult.toString());
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  +/-
export const minus_plus = (displayText,setDisplayText) => {
  const currentText = displayText;
  if (currentText[0] === '-') {
    setDisplayText(currentText.slice(1));
  } else {
    setDisplayText('-' + currentText);
  }
};


//  π
export const displayPi = (setDisplayText) => {
  const piValue = Math.PI;
  setDisplayText(piValue.toString());
};


//  ln
export const ln = (displayText,setDisplayText) => {
  try {
    const currentText = eval(displayText);
    if (currentText <= 0) {
      setDisplayText('Błąd: ln z liczby nieujemnej');
    } else {
      const finalResult = Math.log(currentText);
      setDisplayText(finalResult.toString());
    }
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  log10
export const log10 = (displayText,setDisplayText) => {
  try {
    const currentText = eval(displayText);
    if (currentText <= 0) {
      setDisplayText('Błąd: log10 z liczby nieujemnej');
    } else {
      const finalResult = Math.log10(currentText);
      setDisplayText(finalResult.toString());
    }
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  1/x
export const reciprocal = (displayText,setDisplayText) => {
  try {
    const currentText = eval(displayText);
    if (currentText === 0) {
      setDisplayText('Błąd: Nie można obliczyć odwrotności z liczby zero');
    } else {
      const finalResult = 1 / currentText;
      setDisplayText(finalResult.toString());
    }
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  e
export const displayEuler = (setDisplayText) => {
  const eValue = Math.E;
  setDisplayText(eValue.toString());
};


//  E
export const scientificNotation = (displayText,setDisplayText) => {
  const currentText = displayText;
  const numberValue = parseFloat(currentText);

  if (!isNaN(numberValue)) {
    const scientificValue = numberValue.toExponential();
    setDisplayText(scientificValue);
  } else {
    setDisplayText('Błąd: Nieprawidłowa notacja naukowa');
  }
};


//  10^x
export const powerOfTen = (displayText,setDisplayText) => {
  try {
    const currentText = eval(displayText);
    const finalResult = Math.pow(10, currentText);
    setDisplayText(finalResult.toString());
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  2nd
export const power2nd = (displayText,setDisplayText) => {
  try {
    const currentText = displayText;
    const number = parseFloat(currentText);

    if (!isNaN(number)) {
      const result = Math.pow(number, 2);
      setDisplayText(result.toString());
    } else {
      setDisplayText('Błąd obliczeń');
    }
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  e^x
export const expPower = (displayText,setDisplayText) => {
  try {
    const currentText = displayText;
    const number = parseFloat(currentText);

    if (!isNaN(number)) {
      const result = Math.exp(number);
      setDisplayText(result.toString());
    } else {
      setDisplayText('Błąd obliczeń');
    }
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  Rad
export const degreesToRadians = (displayText,setDisplayText) => {
  try {
    const currentText = eval(displayText);
    const radians = (currentText * Math.PI) / 180;
    setDisplayText(radians.toString());
  } catch (error) {
    setDisplayText('Błąd');
  }
};


//  Rand
export const random = (setDisplayText) => {
  const randomValue = Math.random();
  setDisplayText(randomValue.toString());
};


// m+
export const addToMemory = (displayText, setDisplayText, memory, setMemory) => {
  const currentValue = parseFloat(displayText);
  const currentMemory = parseFloat(memory);

  if (!isNaN(currentValue)) {
    const newMemory = currentMemory + currentValue;
    setMemory(newMemory);
    setDisplayText(`${newMemory}`);
  }
};

// m-
export const subtractFromMemory = (displayText, setDisplayText, memory, setMemory) => {
  const currentValue = parseFloat(displayText);
  const currentMemory = parseFloat(memory);

  if (!isNaN(currentValue)) {
    const newMemory = currentMemory - currentValue;
    setMemory(newMemory);
    setDisplayText(`${newMemory}`);
  }
};

// mr
export const recallFromMemory = (displayText, setDisplayText, memory) => {
  const currentMemory = parseFloat(memory);
  if (!isNaN(currentMemory)) {
    setDisplayText(`${currentMemory}`);
  }
};

// mc
export const clearMemory = (setMemory) => {
  setMemory(0);
};


//  !x
export const factorial = (displayText,setDisplayText) => {
  try {
    const currentText = displayText;
    const number = parseFloat(currentText);

    if (!isNaN(number) && Number.isInteger(number) && number >= 0) {
      const result = math.factorial(number);
      setDisplayText(result.toString());
    } else {
      setDisplayText('Błąd');
    }
  } catch (error) {
    setDisplayText('Błąd');
  }
};
export const roundToTwoDecimalPlaces = () => {
  return Math.round(value * 100) / 100;
};


//  Trigonometric functions
export const trigFunction = (displayText, setDisplayText, funcName) => {
  const input = parseFloat(displayText);
  if (!isNaN(input)) {
    let result;

    switch (funcName) {
      case 'sin':
        result = Math.sin(input);
        break;
      case 'cos':
        result = Math.cos(input);
        break;
      case 'tan':
        result = Math.tan(input);
        break;
      case 'sinh':
        result = Math.sinh(input);
        break;
      case 'cosh':
        result = Math.cosh(input);
        break;
      case 'tanh':
        result = Math.tanh(input);
        break;
      default:
        console.error('Unknown trigonometric function');
        return;
    }
    setDisplayText(result.toString());
  } else {
    console.error('Invalid input for trigonometric function');
  }
};