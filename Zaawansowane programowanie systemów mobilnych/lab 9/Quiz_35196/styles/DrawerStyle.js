// DrawerStyle.js

import { StyleSheet } from 'react-native';


const styles = StyleSheet.create({
  drawerButton: {
    backgroundColor: '#808080',
    paddingVertical: 10,
    paddingHorizontal: 15,
    marginVertical: 10,
    borderRadius: 2,
  },
  drawerButtonText: {
    fontSize: 14,
    textAlign: 'center',
    color: 'white',
  },
  scrollContainer: {
    flexGrow: 1,
  },
  containerDrawer: {
    flex: 1,
    alignItems: 'center',
  },
  navigationContainer: {
    backgroundColor: '#ecf0f1',
    padding: 20,
  },
  drawerTitle: {
    fontSize: 25,
    fontWeight: 'bold',
    textAlign: 'center',
  },
  drawerIcon: {
    width: 150,
    height: 150,
    alignSelf: 'center',
    marginVertical: 20,
  },
  buttonContainer: {
    marginTop: 15,
    width: '100%', 
    flexDirection: 'column',
  },
  buttonSpacer: {
    marginVertical: 10, 
  },
  divider: {
    borderBottomColor: '#ccc',
    borderBottomWidth: 2,
    marginBottom: 10,
    marginTop: 4,
  },
});

export default styles;