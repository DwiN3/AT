// EditDeviceStyle.js

import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  container: {
    padding: 16,
  },
  input: {
    height: 40,
    borderColor: 'gray',
    borderWidth: 1,
    marginBottom: 16,
    paddingLeft: 8,
  },
  colorsContainer: {
    height: 55,
    borderRadius: 20,
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 170,
    backgroundColor: "grey",
  },
  circleOneColor: {
    width: 40,
    height: 40,
    borderRadius: 20,
    borderWidth: 2,
    margin: 7,
  },
  buttonContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  button: {
    borderColor: 'black',
    borderWidth: 2,
    backgroundColor: 'transparent',
    width: '100%',
    height: 40,
    fontWeight: '600',
    alignItems: 'center',
    justifyContent: 'center',
  },
  buttonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
});

export default styles;