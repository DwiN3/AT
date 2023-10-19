import React, { useState } from 'react';
import { Text, View, StyleSheet, TouchableOpacity } from 'react-native';

const App = () => {
  const [isVisible, setIsVisible] = useState(true);
  const [buttonText, setButtonText] = useState('Ukryj');

  const toggleVisibility = () => {
    setIsVisible(!isVisible);
    setButtonText(isVisible ? 'Pokaż' : 'Ukryj');
  };

  return (
    <View style={styles.container}>
      <Text style={styles.header}>Zadanie 2</Text>
      <TouchableOpacity onPress={toggleVisibility}>
        <Text style={styles.buttonText}>{buttonText}</Text>
      </TouchableOpacity>
      {isVisible && (
        <View style={styles.itemContainer}>
          <Text>Nazywam się:</Text>
          <Text style={styles.boldText}>Kamil Dereń</Text>
        </View>
      )}
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  header: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  itemContainer: {
    marginVertical: 10,
  },
  boldText: {
    fontWeight: 'bold',
  },
  buttonText: {
    borderWidth: 1,
    padding: 10,
    borderRadius: 5,
  },
});

export default App;
