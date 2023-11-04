export const handleButtonPress = (buttonValue, displayText, setDisplayText) => {
    if (displayText === '0') {
      setDisplayText(buttonValue);
    } else {
      setDisplayText((prevText) => prevText + buttonValue);
    }
  };
  
  export const clearDisplay = (displayText, setDisplayText) => {
    setDisplayText('0');
  };
  
  export const calculateResult = (displayText, setDisplayText) => {
    try {
      const result = eval(displayText);
      setDisplayText(result.toString());
    } catch (error) {
      setDisplayText('Błąd');
    }
  };
  
  export const power = (displayText, setDisplayText) => {
    try {
      const result = eval(displayText);
      setDisplayText(result.toString());
    } catch (error) {
      setDisplayText('Błąd');
    }
  };
  