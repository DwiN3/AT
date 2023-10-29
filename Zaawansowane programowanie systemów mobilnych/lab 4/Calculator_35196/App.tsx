import React, { useState, useEffect } from 'react';
import { View, TouchableOpacity, Text, StyleSheet, Dimensions } from 'react-native';
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
    Dimensions.addEventListener('change', handleOrientationChange);

    return () => {
      Dimensions.removeEventListener('change', handleOrientationChange);
    };
  }, []);

  return (
    <View style={styles.container}>
      <View style={orientation === 'landscape' ? styles.displayLand : styles.display}>
      <Text style={styles.displayText}>{displayText}</Text>
</View>
      {orientation === 'portrait' ? (
        <View style={[styles.buttons, { flex: 4 }]}>
          <View style={styles.row}>
            <TouchableOpacity style={[styles.buttonClear, { flex: 3 }]} onPress={clearDisplay}>
              <Text style={styles.buttonText}>AC</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('/')}
            >
              <Text style={styles.buttonTextOperator}>/</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('7')}>
              <Text style={styles.buttonText}>7</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('8')}>
              <Text style={styles.buttonText}>8</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('9')}>
              <Text style={styles.buttonText}>9</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('*')}
            >
              <Text style={styles.buttonTextOperator}>*</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('4')}>
              <Text style={styles.buttonText}>4</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('5')}>
              <Text style={styles.buttonText}>5</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('6')}>
              <Text style={styles.buttonText}>6</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('-')}
            >
              <Text style={styles.buttonTextOperator}>-</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('1')}>
              <Text style={styles.buttonText}>1</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('2')}>
              <Text style={styles.buttonText}>2</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('3')}>
              <Text style={styles.buttonText}>3</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('+')}
            >
              <Text style={styles.buttonTextOperator}>+</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
            <TouchableOpacity style={styles.buttonZero} onPress={() => handleButtonPress('0')}>
              <Text style={styles.buttonText}>0</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('.')}>
              <Text style={styles.buttonText}>.</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={calculateResult}
            >
              <Text style={styles.buttonTextOperator}>=</Text>
            </TouchableOpacity>
          </View>
        </View>
      ) : (
        <View style={[styles.buttons, { flex: 10 }]}>
          <View style={styles.row}>
            <TouchableOpacity style={styles.moreButtons} onPress={() => handleButtonPress('(')}>
              <Text style={styles.moreTextButtons}>(</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => handleButtonPress(')')}>
              <Text style={styles.moreTextButtons}>)</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={clearMemory}>
              <Text style={styles.moreTextButtons}>mc</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={addToMemory}>
              <Text style={styles.moreTextButtons}>m+</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={subtractFromMemory}>
              <Text style={styles.moreTextButtons}>m-</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={recallFromMemory}>
              <Text style={styles.moreTextButtons}>mr</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={clearDisplay}>
              <Text style={styles.moreTextButtons}>AC</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={minus_plus}>
              <Text style={styles.moreTextButtons}>+/-</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={procent}>
              <Text style={styles.moreTextButtons}>%</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('/')}
            >
              <Text style={[styles.buttonTextOperator, { fontSize: 25 }]}>/</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
            <TouchableOpacity style={styles.moreButtons} onPress={power2nd}>
              <Text style={styles.moreTextButtons}>2nd</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => power(2)}>
              <Text style={styles.moreTextButtons}>x²</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => power(3)}>
              <Text style={styles.moreTextButtons}>x³</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => power('y')}>
              <Text style={styles.moreTextButtons}>x^y</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={expPower}>
              <Text style={styles.moreTextButtons}>e^x</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={powerOfTen}>
              <Text style={styles.moreTextButtons}>10^x</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('7')}>
              <Text style={styles.moreTextButtons}>7</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('8')}>
              <Text style={styles.moreTextButtons}>8</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('9')}>
              <Text style={styles.moreTextButtons}>9</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('*')}
            >
              <Text style={[styles.buttonTextOperator, { fontSize: 25 }]}>*</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
            <TouchableOpacity style={styles.moreButtons} onPress={reciprocal}>
              <Text style={styles.moreTextButtons}>1/x</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => sqrt('√')}>
  <Text style={styles.moreTextButtons}>√x</Text>
</TouchableOpacity>
<TouchableOpacity style={styles.moreButtons} onPress={() => sqrt('³√')}>
  <Text style={styles.moreTextButtons}>³√x</Text>
 </TouchableOpacity>

            <TouchableOpacity style={styles.moreButtons} onPress={() => sqrt('y√x')}>
              <Text style={styles.moreTextButtons}>y√x</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={ln}>
              <Text style={styles.moreTextButtons}>In</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={log10}>
              <Text style={styles.moreTextButtons}>log10</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('4')}>
              <Text style={styles.moreTextButtons}>4</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('5')}>
              <Text style={styles.moreTextButtons}>5</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('6')}>
              <Text style={styles.moreTextButtons}>6</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('-')}
            >
              <Text style={[styles.buttonTextOperator, { fontSize: 25 }]}>-</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
            <TouchableOpacity style={styles.moreButtons} onPress={factorial}>
              <Text style={styles.moreTextButtons}>x!</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => trigFunction('sin')}>
              <Text style={styles.moreTextButtons}>sin</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => trigFunction('cos')}>
              <Text style={styles.moreTextButtons}>cos</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => trigFunction('tan')}>
              <Text style={styles.moreTextButtons}>tan</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={displayEuler}>
              <Text style={styles.moreTextButtons}>e</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={scientificNotation}>
              <Text style={styles.moreTextButtons}>EE</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('1')}>
              <Text style={styles.moreTextButtons}>1</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('2')}>
              <Text style={styles.moreTextButtons}>2</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress('3')}>
              <Text style={styles.moreTextButtons}>3</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={() => handleButtonPress('+')}
            >
              <Text style={[styles.buttonTextOperator, { fontSize: 25 }]}>+</Text>
            </TouchableOpacity>
          </View>
          <View style={styles.row}>
          <TouchableOpacity style={styles.moreButtons} onPress={degreesToRadians}>
              <Text style={styles.moreTextButtons}>Rad</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => trigFunction('sinh')}>
              <Text style={styles.moreTextButtons}>sinh</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => trigFunction('cosh')}>
              <Text style={styles.moreTextButtons}>cosh</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={() => trigFunction('tanh')}>
              <Text style={styles.moreTextButtons}>tanh</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={displayPi}>
              <Text style={styles.moreTextButtons}>π</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.moreButtons} onPress={random}>
              <Text style={styles.moreTextButtons}>Rand</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonZero} onPress={() => handleButtonPress('0')}>
              <Text style={styles.moreTextButtons}>0</Text>
            </TouchableOpacity>
            <TouchableOpacity style={styles.buttonNumber} onPress={() => handleButtonPress(',')}>
              <Text style={styles.moreTextButtons}>.</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.buttonOperator}
              onPress={calculateResult}
            >
              <Text style={[styles.buttonTextOperator, { fontSize: 25 }]}>=</Text>
            </TouchableOpacity>
          </View>
        </View>
        )}
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
  displayLand: {
    flex: 3,
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
    backgroundColor: 'gray',
  },
  row: {
    flex: 1,
    flexDirection: 'row',
  },
  button: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#434343',
  },
  buttonText: {
    color: 'white',
    fontSize: 35,
  },
  buttonClear: {
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#434343',
    backgroundColor: '#646466',
  },
  moreButtons: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#434343',
    backgroundColor: '#646466',
  },
  buttonTextClear: {
    color: 'white',
    fontSize: 35,
  },
  moreTextButtons: {
    color: 'white',
    fontSize: 25,
  },
  buttonOperator: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#434343',
    backgroundColor: 'orange',
  },
  buttonTextOperator: {
    color: 'white',
    fontSize: 40,
  },
  buttonNumber: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    borderColor: '#434343',
    color: 'white',
    backgroundColor: '#7c7d7f',
  },
  buttonZero: {
    flex: 2,
    justifyContent: 'center',
    alignItems: 'center',
    borderWidth: 1,
    fontSize: 35,
    borderColor: '#7c7d7f',
    backgroundColor: '#7c7d7f',
  }
});
export default App;
