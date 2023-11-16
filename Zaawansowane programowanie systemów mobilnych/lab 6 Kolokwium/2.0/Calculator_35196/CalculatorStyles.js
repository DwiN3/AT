//  CalculatorStyles.js
import { StyleSheet } from 'react-native';

const commonButtonStyle = {
  flex: 1,
  borderWidth: 1,
  borderColor: '#434343',
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#535457',
  },
  display: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'flex-end',
    backgroundColor: '#535457',
    padding: 10,
  },
  displayText: {
    fontSize: 65,
    color: 'white',
  },
  buttons: {
    flex: 4,
    flexDirection: 'column',
  },
  row: {
    flex: 2,
    flexDirection: 'row',
  },
  buttonText: {
    color: 'white',
    fontSize: 10,
    paddingVertical: 20,
    textAlign: 'center',
  },
  buttonMore: {
    ...commonButtonStyle,
    backgroundColor: '#646466',
  },
  buttonOperator: {
    ...commonButtonStyle,
    backgroundColor: 'orange',
  },
  buttonNumber: {
    ...commonButtonStyle,
    backgroundColor: '#7c7d7f',
  },
});

export default styles;
