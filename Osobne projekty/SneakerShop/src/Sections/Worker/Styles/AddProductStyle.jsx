import { StyleSheet, Dimensions } from "react-native";

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;

const primaryColor = '#411c5d';
const secColor = '#694182';

export const styles = StyleSheet.create({
    body: {
        backgroundColor: '#c3c3dc',
        flex: 1, 
        alignItems: 'center',
    },
    keyboardBox: {
        flex: 1, 
    },
    mainContainer:{
        alignItems: 'center',
        width,
    },
    titleText:{
        color: primaryColor,
        fontFamily: 'Mina-Bold',
        fontSize: 40,
        marginVertical: 10,
    },
    inputContainer:{
        backgroundColor: secColor,
        width: width * 0.9,
        height: 50,
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        marginTop: 15,
        alignContent: 'center',
        justifyContent: 'center'
    },
    inputText:{
        marginHorizontal: 20,
        color: 'white',
        fontSize: 16,
        fontFamily: 'Mina-Regular',
    },
    placeholder: {
        color: 'white',
    },
    buttonContainer:{
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: primaryColor,
        width: width * 0.8,
        height: 50,
        borderRadius: 20,
        marginTop: 15,
        marginBottom: 15,
    },
    buttonText:{
        color: 'white',
        fontSize: 25,
        fontWeight: 'bold'
    },
    photoInputContainer: {
        width: width * 0.9,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        height: 55,
        backgroundColor: '#694187',
        borderRadius: 10,
        marginVertical: 10,
    },
    photoPlaceholder: {
        color: '#fff',
        marginLeft: 10,
        fontSize: 18,
        fontFamily: 'Mina-Regular',
    },
    photoContainer: {
        flexDirection: 'row',
        width: width * 0.9,
        marginVertical: 10,
    },
    photoWrapper: {
        position: 'relative',
        marginRight: 10,
    },
    photo: {
        width: 100,
        height: 100,
        borderRadius: 10,
    },
    selectedPhoto: {
        borderWidth: 2,
        borderRadius: 10,
        borderColor: secColor,
    },
    removePhotoIcon: {
        position: 'absolute',
        top: 5,
        right: 5,
        backgroundColor: 'rgba(255, 255, 255, 0.8)',
        borderRadius: 15,
        padding: 2,
    },

    sizeInputContainer: {
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'space-between',
        width: width * 0.9,
        marginVertical: 10,
        gap: 10,
    },
    sizeInput: {
        backgroundColor: '#fff',
        flex: 1,
        height: 40,
        borderRadius: 10,
        paddingHorizontal: 10,
        fontSize: 16,
        color: primaryColor,
        borderWidth: 1,
        borderColor: primaryColor,
    },
    addButton: {
        backgroundColor: primaryColor,
        paddingVertical: 10,
        paddingHorizontal: 20,
        borderRadius: 10,
    },
    addButtonText: {
        color: '#fff',
        fontSize: 16,
    },
    sizeListContainer: {
        width: width * 0.9,
    },
    sizeItem: {
        flexDirection: 'row',
        justifyContent: 'space-between',
        alignItems: 'center',
        backgroundColor: '#fff',
        paddingVertical: 10,
        paddingHorizontal: 20,
        borderRadius: 10,
        marginVertical: 5,
    },
    sizeText: {
        color: primaryColor,
        fontSize: 16,
    },
    errorMessage: {
        alignSelf: 'flex-start',
        marginLeft: '15%',
        fontFamily: 'Mina-Bold',
        fontSize: 14,
        color: 'red',
    },
    pickerContainer: {
        width: width * 0.9,
        flexDirection: 'row',
        alignItems: 'center',
        justifyContent: 'center',
        height: 55,
        backgroundColor: '#694187',
        borderRadius: 10,
        marginVertical: 10,
        paddingLeft: '5%',
    },
    pickerLabel: {
        flex: 1,
        color: '#fff',
        marginLeft: 10,
        fontSize: 18,
        fontFamily: 'Mina-Regular',
    },
    picker: {
        flex: 1.5,
        color: '#fff',
        fontSize: 18,
        fontFamily: 'Mina-Regular',
        backgroundColor: 'rgba(231, 224, 227, 0.21)',
    },
});