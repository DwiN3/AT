// ResultsStyle.js

import { StyleSheet } from 'react-native';

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
  },
  evenRow: {
    backgroundColor: '#fff', 
  },
  oddRow: {
    backgroundColor: '#f2f2f2',
  },
  cell: {
    flex: 1,
    color: 'black',
    textAlign: 'center',
  },
});

export default styles;