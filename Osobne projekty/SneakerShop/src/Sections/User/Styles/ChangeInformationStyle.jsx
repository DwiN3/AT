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
    
    form: {
        width: width - 20,
        marginTop: 15,
        flexGrow: 1,
    },

    card: {
        backgroundColor: '#fff',
        width: width - 25,
        borderRadius: 20,
    },

    header: {
        height: height /5,
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'space-between',
    },

    headerTextBox: {
        flex: 1.2,
        backgroundColor: purple,
        marginVertical: 20,
        borderTopRightRadius: 20,
        borderBottomRightRadius: 20,
        alignContent: 'center',
        justifyContent: 'center'
    },

    stamp: {
        flex: 1,
        height: '85%',
        objectFit: 'contain',
        alignSelf: 'center',
        marginVertical: 20,
        marginHorizontal: 10
    },

    title: {
        color: '#fff',
        textAlign: 'center',
        fontSize: 25,
        fontFamily: 'Mina-Bold'
    },

    /* Form */
    inputBackground: {
        borderBottomWidth: 1,
        borderBottomColor: '#676e78',
        width: '85%',
        alignSelf: 'center',
    },

    input: {
        height: 38,
        fontSize: 15,
        textAlignVertical: 'bottom',
        paddingHorizontal: 10,
    },

    inputError: {
        color: 'red'
    },

    button: {
        width: width * 0.7,
        height: 50,
        backgroundColor: purple,
        borderRadius: 20,
        alignSelf: 'center',
        margin: 25,
        alignContent: 'center',
        justifyContent: 'center'
    },

    errorMessageBox: {
        paddingHorizontal: 20,
        fontFamily: 'Mina-Regular'
    },

    errorMessage: {
        fontFamily: 'Mina-Regular',
        color: 'red'
    },

    textHint: {
        marginTop: 20,
        color: purple,
        fontFamily: 'Mina-Bold',
        fontSize: 16
    },

    /* Newsletter */
    contentNewsletter: {
        flexGrow: 1,
        alignItems: 'center',
        justifyContent: 'center'
    },

    titleNewsletter: {
        color: '#fff',
        textAlign: 'center',
        fontSize: 25,
        fontFamily: 'Mina-Regular',
        marginBottom: 25
    },

    customRadioButton: {
        width: width* 0.9,
        height: 50,
        borderWidth: 2,
        borderRadius: 20,
        marginVertical: 5,
        alignItems: 'center',
        justifyContent: 'center',
    },

    customRadioButtonNormal: {
        backgroundColor: 'rgba(255 255 255 / 0.6)',
        borderColor: purple,
    },

    customRadioButtonSelected: {
        backgroundColor: purple,
        borderColor: '#fff',
    },

    customRadioButtonText: {
        color: purple,
        fontFamily: 'Mina-Regular',
        fontSize: 16
    },

    customRadioButtonTextSelected: {
        color: '#fff',
        fontFamily: 'Mina-Regular',
        fontSize: 16
    }
});