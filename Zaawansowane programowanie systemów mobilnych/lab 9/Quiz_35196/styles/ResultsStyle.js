// ResultsStyle.js

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
  scrollContainer: {
    padding: 0,
  },
  row: {
    flexDirection: 'row',
    borderBottomWidth: 1,
    borderColor: '#ccc',
    paddingVertical: 8,
  },
  header: {
    backgroundColor: '#808080',
    fontFamily: 'roboto-regular',
  },
  evenRow: {
    backgroundColor: '#fff',
  },
  oddRow: {
    backgroundColor: '#f2f2f2',
  },
  headerCell: {
    flex: 1,
    color: 'black',
    textAlign: 'center',
    fontFamily: 'roboto-bold',
    padding: 8, 
  },
  cell: {
    flex: 1,
    color: 'black',
    textAlign: 'center',
    fontFamily: 'roboto-regular',
    padding: 8, 
  },
});

export default styles;
