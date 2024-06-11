import { StyleSheet } from "react-native";
import {Dimensions} from 'react-native';

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;

export const styles = StyleSheet.create({
    body: {
        width,
        height,
        backgroundColor: '#C3C7DF'
    },
    background: {
        position: 'absolute',
        width,
        height
    },

    /* Header */
    header: {
        height: height/2.2,
    },
    iconBox: {
        display: 'flex',
        flexDirection: 'row',
        marginTop: -60,
        justifyContent: 'space-between',
        marginHorizontal: 2,
    },
    icon: {
        width: width/2,
        height: 140,
    },
    logo: {
        width: width/1.3,
        objectFit: 'contain',
        alignSelf: 'center',
    },
    title: {
        marginTop: -10,
        color: '#fff',
        textAlign: 'center',
        fontSize: 60,
        fontFamily: 'Mina-Bold'
    },

    /* Login form */
    form: {
        flex: 1,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },

    buttonBox: {
        width: '75%',
        height: 55,
        marginVertical: 10,
        backgroundColor: '#490e75',
        borderRadius: 15
    },
    button: {
        width: '100%',
        height: '100%',
        justifyContent: 'center',
        alignItems: 'center'
    },
    buttonText: {
        fontSize: 30,
        color: '#fff',
        fontFamily: 'Mina-Bold'
    },
    signUp: {
        display: 'flex',
        flexDirection: 'row',
    },
    signUpText: {
        color: '#000',
        fontFamily: 'Mina-Regular'
    },
    signUpLink: {
        color: '#490e75',
        fontFamily: 'Mina-Bold'
    }
});