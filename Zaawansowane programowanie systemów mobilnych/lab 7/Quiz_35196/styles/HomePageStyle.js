// HomePageStyle.js

import { StyleSheet } from 'react-native';

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
    fontSize: 22,
    fontWeight: 'bold',
    marginBottom: 5,
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