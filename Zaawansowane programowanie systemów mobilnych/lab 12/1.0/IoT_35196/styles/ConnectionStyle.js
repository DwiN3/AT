// ConnectionStyle.js
import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  buttonContainer: {
    marginTop: 10,
    marginLeft: 20,
    marginRight: 20,
  },
  button: {
    padding: 10,
    alignItems: 'center',
    borderRadius: 5,
    marginBottom: 10,
  },
  buttonScan: {
    marginTop: -20,
    padding: 20,
    alignItems: 'center',
    borderRadius: 5,
    marginBottom: 50,
    marginHorizontal: 0,
    backgroundColor: 'gray',
  },
  redButton: {
    backgroundColor: 'red',
  },
  greenButton: {
    backgroundColor: 'green',
  },
  blueButton: {
    backgroundColor: 'blue',
  },
  offButton: {
    backgroundColor: 'gray',
  },
  buttonText: {
    color: 'white',
  },
});

export default styles;
