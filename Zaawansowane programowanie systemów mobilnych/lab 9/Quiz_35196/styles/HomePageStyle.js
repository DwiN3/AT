// HomePageStyle.js

import { StyleSheet } from 'react-native';
import * as Font from 'expo-font';

const getFonts = async () => {
  await Font.loadAsync({
    'oswald-regular': require('../assets/fonts/Oswald-Regular.ttf'),
    'oswald-bold': require('../assets/fonts/Oswald-Bold.ttf'),
    'roboto-regular': require('../assets/fonts/Roboto-Regular.ttf'),
    'roboto-bold': require('../assets/fonts/Roboto-Bold.ttf'),
  });
};

getFonts();

const styles = StyleSheet.create({
  resultsItem: {
    width: '111%',
    height: 120,
    borderWidth: 2,
    borderColor: 'black',
    justifyContent: 'center',
    alignItems: 'center',
    borderRadius: 5,
    marginTop: 10,
    marginBottom: -14,
    marginLeft: -21, 
    marginRight: -10, 
  },
  resultsItemText: {
    fontSize: 18,
    fontWeight: 'bold',
  },
  resultsItemButton: {
    backgroundColor: 'grey',
    paddingVertical: 10,
    paddingHorizontal: 15,
    marginVertical: 10,
    borderRadius: 2,
  },
  resultsItemButtonText: {
    fontSize: 14,
    textAlign: 'center',
    color: 'white',
  },
  testItem: {
    marginVertical: 10,
    padding: 15,
    width: '100%',
    height: 150, 
    borderWidth: 3,
    borderColor: '#ccc',
    borderRadius: 5,
  },
  regularItem: {
    marginBottom: 10,
  },
  titleTest: {
    marginTop: -9,
    fontSize: 22,
    marginBottom: 5,
    fontFamily: 'oswald-bold',
  },
  tagsContainer: {
    flexDirection: 'row',
    marginBottom: 5,
  },
  tag: {
    fontSize: 14,
    marginRight: 5,
    color: 'blue',
  },
  description: {
    marginTop: 11,
    fontSize: 14,
  },
  containerHome: {
    flex: 1,
    alignItems: 'center',
  },
  flatListContainer: {
    flexGrow: 1,
    padding: 20,
  },
});

export default styles;