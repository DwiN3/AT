// DeviceStyle.js

import { StyleSheet } from 'react-native';

const styles = StyleSheet.create({
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