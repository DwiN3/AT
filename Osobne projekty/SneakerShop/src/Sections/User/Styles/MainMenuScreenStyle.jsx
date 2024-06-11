import { StyleSheet, Dimensions } from "react-native";

const primaryColor = '#411c5d';

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;

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
        flexGrow: 1,
        flexDirection: 'row',
        flexWrap: 'wrap',
        rowGap: 15,
        columnGap: 15,
        justifyContent: 'space-around',
        alignItems: 'flex-start',
        padding: 10,
        paddingBottom: 20
    },
    productContainer:{
        backgroundColor: "#d7d7d7",
        width: (width - 35) /2,
        height: 220,
        borderColor: primaryColor,
        borderWidth: 2,
        borderRadius: 20,
        justifyContent: 'center',
        alignItems: 'center',
    },
    productImage:{
        flex: 1,
        height: '100%',
        width: '100%',
        marginBottom: 70,
        borderTopLeftRadius: 20,
        borderTopRightRadius: 20,
    },
    infoProductContainer:{
        backgroundColor: "white",
        width: "103%",
        height: 90,
        borderColor: primaryColor,
        borderWidth: 2,
        borderRadius: 20,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'absolute',
        gap: 0,
        bottom: -4,
        padding: 3
    },
    infoProductText:{
        color: primaryColor,
        fontSize: 18,
        fontFamily: 'Mina-Bold',
        textAlign: 'center'
    },
    backgroundFilter:{
        width: "100%",
        height: "100%",
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'absolute',
    },
    filterContainer:{
        backgroundColor: "white",
        width: width * 0.85,
        height: height * 0.85,
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },
    filterContentContainer:{
        display: 'flex',
        alignItems: 'center',
        backgroundColor: "white",
        width: 322,
    },
    buttonFilterContainer: {
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
    },
    buttonFilter:{
        marginHorizontal: 10,
        backgroundColor: primaryColor,
        width: 120,
        height: 45,
        borderRadius: 10,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        marginTop: 10,
        marginBottom: 10,
    },
    textButtonFilter:{
        color: 'white',
        fontFamily: 'Mina-Bold',
        fontSize: 17,
    },
    closeImage:{
        width: 60,
        height: 40,
        right: -65,
        marginTop: 5,
        marginBottom: 5,
        marginRight: 5,
    },
    checkboxContainer: {
        display: 'flex',
        flexDirection: 'column',
    },
    checkbox: {
        width: 200,
        height: 30,
        display: 'flex',
        flexDirection: 'row',
        alignItems: 'center',
        marginBottom: 5,
    },
    checkboxIcon: {
        width: 24,
        height: 24,
        borderWidth: 2,
        borderColor: '#000',
        marginRight: 10,
    },
    checkedIcon: {
        backgroundColor: primaryColor, 
    },
    titleText:{
        fontSize: 20,
        fontFamily: 'Mina-Bold',
        marginLeft: 10,
        marginBottom: 10,
        color: primaryColor
    },
    checkboxText:{
        fontFamily: 'Mina-Bold',
        fontSize: 16,
        marginLeft: 15,
    },
    borderFilder: {
        marginTop: 15,
        marginBottom: 15,
        width: 250,
        height: 2,
        backgroundColor: primaryColor,
    },
    inputSection: {
        display: 'flex',
        flexDirection: 'column',
    },
    inputContainers: {
        display: 'flex',
        flexDirection: 'row',
        justifyContent:"space-around",
        gap: 20,
        width: 200,
    },
    inputName:{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        width: 220,
        height: 50,
        borderWidth: 1,
        borderRadius: 10,
    },
    input:{
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        width: 100,
        height: 50,
        borderWidth: 1,
        borderRadius: 10,
    },
    inputText:{
        width: '100%',
        textAlign: 'center',
        color: primaryColor,
        fontSize: 16,
        fontFamily: 'Mina-Bold'
    },
    placeholderText: {
        color: primaryColor,
    },
    sortContent: {
        width,
        height: 225,
        backgroundColor: 'rgba(255, 255, 255, 0.8)',        
    },
    sortTitle: {
        width,
        height: 60,
        fontFamily: 'Mina-Regular',
        fontSize: 20,
        color: primaryColor,
        textAlign: 'center',
        paddingTop: 15
    },
    sortButton: {
        height: 40,
        width,
        justifyContent: 'space-between',
        alignItems: 'center',
        flexDirection: 'row',
        borderBottomColor: 'gray',
        borderBottomWidth: 1,
        paddingHorizontal: 15
    },
    sortButtonChoosen: {
        borderBottomColor: primaryColor,
        borderBottomWidth: 2,
    },
    sortButtonText: {
        color: 'gray',
        fontFamily: 'Mina-Regular',
        fontSize: 15
    },
    sortButtonTextChoosen: {
        fontFamily: 'Mina-Bold',
        color: primaryColor
    }
});
