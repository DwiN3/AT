import { StyleSheet } from "react-native";
import {Dimensions} from 'react-native';

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;

const purple = '#490e75';

export const styles = StyleSheet.create({
    body: {
        backgroundColor: '#c3c3dc',
        alignItems: 'center',
        flex: 1,
    },

    content: {
        flexGrow: 1,
    },

    header: {
        height: height * 0.3,
        width,
    },

    headerBox: {
        backgroundColor: purple,
        marginVertical: 20,
        height: 110,
        width: '85%',
        borderTopRightRadius: 55,
        borderBottomRightRadius: 55,
        alignContent: 'center',
        justifyContent: 'center',
        flexDirection: 'row',
    },

    headerTextBox: {
        alignItems: 'center',
        justifyContent: 'center',
        flexGrow: 1
    },

    title: {
        color: '#fff',
        textAlign: 'center',
        fontSize: 25,
        fontFamily: 'Mina-Bold'
    },

    email: {
        color: '#fff',
        textAlign: 'center',
        fontSize: 16,
        fontFamily: 'Mina-Regular',
    },

    person: {
        height: 110,
        width: 110,
        borderRadius: 55,
        borderWidth: 3,
        borderColor: purple,
        marginBottom: 5,
        backgroundColor: '#fff'
    },

    personImage: {
        height: 100,
        width: 100,
        objectFit: 'cover',
        borderRadius: 50
    },

    headerTextDescription: {
        paddingHorizontal: 25,
        fontFamily: 'Mina-Regular',
        color: 'purple',
        textAlign: 'center',
    },
    
    button: {
        width,
        height: 50,
        borderBottomColor: '#fff',
        borderBottomWidth: 1,
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        paddingHorizontal: 20,
    },

    buttonText: {
        fontFamily: 'Mina-Regular',
        color: '#fff',
        fontSize: 18,
    },
});