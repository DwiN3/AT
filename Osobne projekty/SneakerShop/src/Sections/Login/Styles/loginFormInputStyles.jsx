import { StyleSheet } from "react-native";

export const styles = StyleSheet.create({
    inputBox: {
        width: '90%',
        height: 55,
        marginVertical: 5,
        backgroundColor: '#694187',
        borderRadius: 15,
        paddingHorizontal: '5%',
        flexDirection: 'row',
        alignItems: 'center',
    },
    inputText: {
        flex: 1,
        marginHorizontal: '2%',
        color: '#fff',
        fontSize: 18,
        fontFamily: 'Mina-Regular',
        textAlignVertical: 'bottom',
        paddingVertical: 0
    },
    markText: {
        color: 'red'
    },
    errorMessage: {
        fontFamily: 'Mina-Bold',
        fontSize: 14,
        color: 'red',
    },
    errorMessageBox: {
        alignSelf: 'flex-start',
        marginLeft: '10%'
    }
});