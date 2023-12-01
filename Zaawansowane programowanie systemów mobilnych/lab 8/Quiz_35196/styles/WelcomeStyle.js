// WelcomeStyle.js

import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    borderColor: '#000',   
    borderStyle: 'solid',  
    padding: 10,
    width: '100%', 
  },
  label: {
    fontWeight: 'bold',
  },
  checkboxContainer: {
    marginTop: 20,
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 20,
  },
  checkbox: {
    width: 20,
    height: 20,
    borderRadius: 5,
    borderWidth: 1,
    borderColor: '#000',
    marginRight: 10,
  },
  checkedBox: {
    backgroundColor: '#3498db',
  },
  continueButton: {
    marginTop: 'auto', 
    marginBottom: 20,  
    padding: 10,
    backgroundColor: '#3498db',
    borderRadius: 5,
    alignItems: 'center',
  },
  disabledButton: {
    backgroundColor: '#808080',
  },
});

export default styles;