import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet, Dimensions } from 'react-native';
import SplashScreen from 'react-native-splash-screen'
const math = require('mathjs');

const App = () => {
  const [displayText, setDisplayText] = useState('0');
  const [orientation, setOrientation] = useState(
    Dimensions.get('window').width > Dimensions.get('window').height ? 'landscape' : 'portrait'
  );

  const handleButtonPress = (buttonValue) => {
    if (displayText === '0' && /[0-9]/.test(buttonValue)) {
      setDisplayText(buttonValue);
    } else {
      setDisplayText((prevText) => {
        if (prevText === '0' && buttonValue === '(') {
          return buttonValue;
        } else {
          return prevText + buttonValue;
        }
      });
    }
  };

  // AC
  const clearDisplay = () => {
    setDisplayText('0');
  };

  // =
  const calculateResult = () => {
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
        result = eval(text);
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
  
  
  
  // ^
  const power = (exponent: number | string) => {
    try {
      let result;
      if (exponent === 'y') {
        setDisplayText(displayText.toString() + "^" );
      } else {
        const base = parseFloat(displayText);
        const exp = parseFloat(exponent as string);
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
      }
    } catch (error) {
      setDisplayText('Błąd');
    }
  };
  
  // √
  const sqrt = (exponent: number | string) => {
    try {
      if (exponent === '√') {
        const x = parseFloat(displayText);
        if (!isNaN(x)) {
          const result = Math.sqrt(x);
          if (!isNaN(result)) {
            setDisplayText(result.toString());
          } else {
            setDisplayText('Błąd obliczeń');
          }
        }
      } else if (exponent === '³√') {
        const x = parseFloat(displayText);
        if (!isNaN(x)) {
          const result = Math.cbrt(x);
          if (!isNaN(result)) {
            setDisplayText(result.toString());
          } else {
            setDisplayText('Błąd obliczeń');
          }
        }
      } else if (exponent === 'y√x') {
        setDisplayText(displayText.toString() + "√");
      }
    } catch (error) {
      setDisplayText('Błąd');
    }
  };
  
  // %
  const procent = () => {
    try {
      const currentText = eval(displayText);
      const finalResult = currentText / 100;
      setDisplayText(finalResult.toString());
    } catch (error) {
      setDisplayText('Błąd');
    }
  };

  // +/-
  const minus_plus = () => {
    const currentText = displayText;
    if (currentText[0] === '-') {
      setDisplayText(currentText.slice(1));
    } else {
      setDisplayText('-' + currentText);
    }
  };

  // π
  const displayPi = () => {
    const piValue = Math.PI;
    setDisplayText(piValue.toString());
  };
  
  // ln
  const ln = () => {
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

  // log10
  const log10 = () => {
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
  
  // 1/x
  const reciprocal = () => {
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

  // e
  const displayEuler = () => {
    const eValue = Math.E;
    setDisplayText(eValue.toString());
  };
  
  // E
  const scientificNotation = () => {
    const currentText = displayText;
    const numberValue = parseFloat(currentText);
  
    if (!isNaN(numberValue)) {
      const scientificValue = numberValue.toExponential();
      setDisplayText(scientificValue);
    } else {
      setDisplayText('Błąd: Nieprawidłowa notacja naukowa');
    }
  };

  // 10^x
  const powerOfTen = () => {
    try {
      const currentText = eval(displayText);
      const finalResult = Math.pow(10, currentText);
      setDisplayText(finalResult.toString());
    } catch (error) {
      setDisplayText('Błąd');
    }
  };

  // 2nd
  const power2nd = () => {
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

  // e^x
  const expPower = () => {
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
  
  // Rad
  const degreesToRadians = () => {
    try {
      const currentText = eval(displayText);
      const radians = (currentText * Math.PI) / 180;
      setDisplayText(radians.toString());
    } catch (error) {
      setDisplayText('Błąd');
    }
  };

  // Rand
  const random = () => {
    const randomValue = Math.random();
    setDisplayText(randomValue.toString());
  };
  
  // Memory functions
  const [memory, setMemory] = useState(0);

  // m+
  const addToMemory = () => {
    try {
      const currentText = parseFloat(displayText);
      setMemory((prevMemory) => prevMemory + currentText);
      setDisplayText('0');
    } catch (error) {
      setDisplayText('Błąd');
    }
  };
  // m-
  const subtractFromMemory = () => {
    try {
      const currentText = parseFloat(displayText);
      setMemory((prevMemory) => prevMemory - currentText);
      setDisplayText('0');
    } catch (error) {
      setDisplayText('Błąd');
    }
  };

  // mr
  const recallFromMemory = () => {
    if (displayText === '0') {
      setDisplayText(memory.toString());
    } else {
      setDisplayText(displayText + memory.toString());
    }
  };
  
  // mc
  const clearMemory = () => {
    setMemory(0);
  };
  
  // !x
  const factorial = () => {
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
  const roundToTwoDecimalPlaces = (value: number) => {
    return Math.round(value * 100) / 100;
  };
  
  // Trigonometric functions
  const toRadians = (degrees: number) => {
    return degrees * (Math.PI / 180);
  };
  const trigFunction = (funcName: string) => {
    try {
      let result;
      const degrees = parseFloat(displayText);
      if (!isNaN(degrees)) {
        const radians = toRadians(degrees);
        switch (funcName) {
          case 'sin':
            result = Math.sin(radians);
            break;
          case 'cos':
            result = Math.cos(radians);
            break;
          case 'tan':
            result = Math.tan(radians);
            break;
          case 'sinh':
            result = Math.sinh(radians);
            break;
          case 'cosh':
            result = Math.cosh(radians);
            break;
          case 'tanh':
            result = Math.tanh(radians);
            break;
          default:
            break;
        }
      }
  
      if (result !== undefined) {
        setDisplayText(roundToTwoDecimalPlaces(result).toString());
      } else {
        setDisplayText('Błąd obliczeń');
      }
    } catch (error) {
      setDisplayText('Błąd');
    }
  };
  
  const handleOrientationChange = () => {
    const { width, height } = Dimensions.get('window');
    if (width > height) {
      setOrientation('landscape');
    } else {
      setOrientation('portrait');
    }
  };

  useEffect(() => {
    if(Platform.OS=== 'android') SplashScreen.hide();
    
    Dimensions.addEventListener('change', handleOrientationChange);

    return () => {
      Dimensions.removeEventListener('change', handleOrientationChange);
    };
  }, []);

  const buttonsPortrait = [
    { title: 'AC', onPress: clearDisplay, style: styles.buttonClear },
    { title: '', style: styles.buttonClear },
    { title: '', style: styles.buttonClear },
    { title: '/', onPress: () => handleButtonPress('/'), style: styles.buttonOperator },
    { title: '7', onPress: () => handleButtonPress('7'), style: styles.buttonNumber },
    { title: '8', onPress: () => handleButtonPress('8'), style: styles.buttonNumber },
    { title: '9', onPress: () => handleButtonPress('9'), style: styles.buttonNumber },
    { title: '*', onPress: () => handleButtonPress('*'), style: styles.buttonOperator },
    { title: '4', onPress: () => handleButtonPress('4'), style: styles.buttonNumber },
    { title: '5', onPress: () => handleButtonPress('5'), style: styles.buttonNumber },
    { title: '6', onPress: () => handleButtonPress('6'), style: styles.buttonNumber },
    { title: '-', onPress: () => handleButtonPress('-'), style: styles.buttonOperator },
    { title: '1', onPress: () => handleButtonPress('1'), style: styles.buttonNumber },
    { title: '2', onPress: () => handleButtonPress('2'), style: styles.buttonNumber },
    { title: '3', onPress: () => handleButtonPress('3'), style: styles.buttonNumber },
    { title: '+', onPress: () => handleButtonPress('+'), style: styles.buttonOperator },
    { title: '0', onPress: () => handleButtonPress('0'), style: [ styles.buttonZero, { flex: 2 }]},
    { title: '.', onPress: () => handleButtonPress('.'), style: styles.buttonNumber },
    { title: '=', onPress: calculateResult, style: styles.buttonOperator },
  ];

  const buttonsLand = [
    { title: '(', onPress: () => handleButtonPress('('), style: styles.buttonClear },
    { title: ')', onPress: () => handleButtonPress(')'), style: styles.buttonClear },
    { title: 'mc', onPress: () => clearMemory(), style: styles.buttonClear },
    { title: 'm+', onPress: () => addToMemory(), style: styles.buttonClear },
    { title: 'm-', onPress: () => subtractFromMemory(), style: styles.buttonClear },
    { title: 'mr', onPress: () => recallFromMemory(), style: styles.buttonClear },
    { title: 'AC', onPress: () => clearDisplay(), style: styles.buttonClear },
    { title: '+/-', onPress: () => minus_plus(), style: styles.buttonClear },
    { title: '%', onPress: () => procent(), style: styles.buttonClear },
    { title: '/', onPress: () => handleButtonPress('/'), style: styles.buttonClear },

    { title: '2nd', onPress: () => power2nd(), style: styles.buttonClear },
    { title: 'x²', onPress: () => power(2), style: styles.buttonClear },
    { title: 'x³', onPress: () => power(3), style: styles.buttonClear },
    { title: 'x^y', onPress: () => power('y'), style: styles.buttonClear },
    { title: 'e^x', onPress: () => expPower(), style: styles.buttonClear },
    { title: '10^x', onPress: () => powerOfTen(), style: styles.buttonClear },
    { title: '7', onPress: () => handleButtonPress('7'), style: styles.buttonNumber },
    { title: '8', onPress: () => handleButtonPress('8'), style: styles.buttonNumber },
    { title: '9', onPress: () => handleButtonPress('9'), style: styles.buttonNumber },
    { title: '*', onPress: () => handleButtonPress('*'), style: styles.buttonOperator },

    { title: '1/x', onPress: () => reciprocal(), style: styles.buttonClear },
    { title: '√x', onPress: () => sqrt('√'), style: styles.buttonClear },
    { title: '³√x', onPress: () => sqrt('³√'), style: styles.buttonClear },
    { title: 'y√x', onPress: () => sqrt('y√x'), style: styles.buttonClear },
    { title: 'In', onPress: () => ln(), style: styles.buttonClear },
    { title: 'log10', onPress: () => log10(), style: styles.buttonClear },
    { title: '4', onPress: () => handleButtonPress('4'), style: styles.buttonNumber },
    { title: '5', onPress: () => handleButtonPress('5'), style: styles.buttonNumber },
    { title: '6', onPress: () => handleButtonPress('6'), style: styles.buttonNumber },
    { title: '-', onPress: () => handleButtonPress('-'), style: styles.buttonOperator },

    { title: 'x!', onPress: () => factorial(), style: styles.buttonClear },
    { title: 'sin', onPress: () => trigFunction('sin'), style: styles.buttonClear },
    { title: 'cos', onPress: () => trigFunction('cos'), style: styles.buttonClear },
    { title: 'tan', onPress: () => trigFunction('tan'), style: styles.buttonClear },
    { title: 'e', onPress: () => displayEuler(), style: styles.buttonClear },
    { title: 'EE', onPress: () => scientificNotation(), style: styles.buttonClear },
    { title: '1', onPress: () => handleButtonPress('1'), style: styles.buttonNumber },
    { title: '2', onPress: () => handleButtonPress('2'), style: styles.buttonNumber },
    { title: '3', onPress: () => handleButtonPress('3'), style: styles.buttonNumber },
    { title: '+', onPress: () => handleButtonPress('+'), style: styles.buttonOperator },

    { title: 'Rad', onPress: () => degreesToRadians(), style: styles.buttonClear },
    { title: 'sinh', onPress: () => trigFunction('sinh'), style: styles.buttonClear },
    { title: 'cosh', onPress: () => trigFunction('cosh'), style: styles.buttonClear },
    { title: 'tanh', onPress: () => trigFunction('tanh'), style: styles.buttonClear },
    { title: 'π', onPress: () => displayPi(), style: styles.buttonClear },
    { title: 'Rand', onPress: () => random(), style:[ styles.buttonClear, { flex:1 } ]},
    { title: '0', onPress: () => handleButtonPress('0'), style: [styles.buttonZero, { flex: 2 }],},
    { title: '.', onPress: () => handleButtonPress('.'), style: styles.buttonNumber },
    { title: '=', onPress: () => calculateResult(), style: styles.buttonOperator },
  ];
  const buttonsToRender = orientation === 'landscape' ? buttonsLand : buttonsPortrait;
  const sliceCount = orientation === 'landscape' ? 10 : 4;
  const sizeFont = orientation === 'landscape'
  ? { ...styles.buttonText, fontSize: 15 }
  : { ...styles.buttonText, fontSize: 35 };

  return (
    <View style={styles.container}>
      <View style={styles.display}>
        <Text style={styles.displayText}>{displayText}</Text>
      </View>
      <View style={styles.buttons}>
        {[0, 1, 2, 3, 4].map((row) => (
          <View key={row} style={styles.row}>
            {buttonsToRender.slice(row * sliceCount, (row + 1) * sliceCount).map((button, index) => (
              <TouchableOpacity
                key={index}
                style={button.style}
                onPress={button.onPress}
              >
                <Text style={sizeFont}>{button.title}</Text>
              </TouchableOpacity>
            ))}
          </View>
        ))}
      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#535457',
  },
  display: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'flex-end',
    backgroundColor: '#535457',
    padding: 10,
  },
  displayText: {
    fontSize: 65,
    color: 'white',
  },
  buttons: {
    flex: 4,
    flexDirection: 'column',
  },
  row: {
    flex: 2,
    flexDirection: 'row',
  },
  buttonText: {
    color: 'white',
    fontSize: 10,
    paddingVertical: 20,
    textAlign: 'center',
  },
  buttonClear: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#434343',
    backgroundColor: '#646466',
  },
  buttonOperator: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#434343',
    backgroundColor: 'orange',
  },
  buttonNumber: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#434343',
    color: 'white',
    backgroundColor: '#7c7d7f',
  },
  buttonZero: {
    flex: 1,
    borderWidth: 1,
    borderColor: '#7c7d7f',
    backgroundColor: '#7c7d7f',
  },
});

export default App;
