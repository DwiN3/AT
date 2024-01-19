// ConnectionStyle.js
import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
  box: {
    backgroundColor: 'grey',
  },
  buttonContainer: {
    marginTop: 10,
    marginLeft: 20,
    marginRight: 20,
  },
  buttonCommends: {
    width: 80,
    height: 50,
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
  commandButtonsContainer: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginTop: -10,
    marginBottom: 10,
  },

  container: {
    flexGrow: 1,
    padding: 10,
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'space-between',
  },
  itemContainer: {
    height: 170,
    width: 170,
    borderWidth: 2,
    borderColor: 'black',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 5,
    marginTop: 10,
    marginLeft: 10,
    marginRight: 10,
    backgroundColor: 'yellow',
    marginBottom: 10,
  },
  deleteItemX: {
    width: 38,
    position: 'absolute',
    top: -59,
    right: -83,
  },
  lastItemContainer: {
    height: 170,
    width: 170,
    borderWidth: 2,
    borderColor: 'black',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 5,
    marginTop: 10,
    marginLeft: 10,
    marginRight: 10,
    marginBottom: 10,
    backgroundColor: 'white', 
  },
  lastItemName: {
    fontSize: 70,
    fontWeight: 'bold',
    color: 'grey',
  },
  itemName: {
    fontSize: 20,
    fontWeight: 'bold',
    color: 'black',
  },
  itemText: {
    fontSize: 16,
    color: '#333',
  },
});

export default styles;
