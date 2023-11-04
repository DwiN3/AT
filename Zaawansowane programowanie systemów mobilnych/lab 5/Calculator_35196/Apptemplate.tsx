const buttons = [
    {
        backgroundColor: '#636466'
        title: 'AC',
        disable: false,
        borderColor: '#555759'
        onPress: () => {

        },
        color: '#E8E9EA'
    },
    {
        backgroundColor: '#636466'
        title: '',
        disable: true,
        borderColor: 'transparent',
        onPress: () => {},
        color: 'transparent'
    },
    {
        backgroundColor: '#636466'
        title: ',',
        disable: true,
        borderColor: '#555759',
        onPress: () => {},
        color: '#E8E9EA'
    },
    {
        backgroundColor: '#636466'
        title: '=',
        disable: false,
        borderColor: '#555759'
        onPress: () => { },
        color: '#E8E9EA'
    }
]

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
    displayLand: {
      flex: 3,
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
      backgroundColor: 'gray',
    },
    row: {
      flex: 1,
      flexDirection: 'row',
    },
    button: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      borderWidth: 1,
      borderColor: '#434343',
    },
    buttonText: {
      color: 'white',
      fontSize: 35,
    },
    buttonClear: {
      justifyContent: 'center',
      alignItems: 'center',
      borderWidth: 1,
      borderColor: '#434343',
      backgroundColor: '#646466',
    },
    moreButtons: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      borderWidth: 1,
      borderColor: '#434343',
      backgroundColor: '#646466',
    },
    buttonTextClear: {
      color: 'white',
      fontSize: 35,
    },
    moreTextButtons: {
      color: 'white',
      fontSize: 25,
    },
    buttonOperator: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      borderWidth: 1,
      borderColor: '#434343',
      backgroundColor: 'orange',
    },
    buttonTextOperator: {
      color: 'white',
      fontSize: 40,
    },
    buttonNumber: {
      flex: 1,
      justifyContent: 'center',
      alignItems: 'center',
      borderWidth: 1,
      borderColor: '#434343',
      color: 'white',
      backgroundColor: '#7c7d7f',
    },
    buttonZero: {
      flex: 2,
      justifyContent: 'center',
      alignItems: 'center',
      borderWidth: 1,
      fontSize: 35,
      borderColor: '#7c7d7f',
      backgroundColor: '#7c7d7f',
    }
  });