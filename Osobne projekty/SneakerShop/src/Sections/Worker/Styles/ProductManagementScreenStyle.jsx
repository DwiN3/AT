import { StyleSheet, Dimensions } from "react-native";

const { width } = Dimensions.get('window');

const primaryColor = '#411c5d';

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
        alignItems: 'center',        
    },
    product:{
        width: width * 0.95,
        display: "flex",
        flexDirection: "column",
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        backgroundColor: "white",
        marginTop: 15
    },
    productData: {
        display: "flex",
        flexDirection: "row",
    },
    productImageContainer:{
        backgroundColor: "#d7d7d7",
        width: 130,
        height: 130,
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        left: -3,
        top: -3
    },
    productImage:{
        flex: 1,
        borderRadius: 20,
        width: '100%',
        height: '100%',
    },
    productInfoContainer:{
        flexGrow: 1,
        display: 'flex',
        alignItems: 'center',
        marginTop: 5,
    },
    nameProductText:{
        color: primaryColor,
        fontSize: 25,
        fontFamily: 'Mina-Bold',
        width: (width * 0.95) - 140,
        textAlign: 'center'
    },
    sizeContainer:{
        width: 160
    },
    tableHeader: {
        color: '#000',
        fontFamily: 'Mina-Bold',
        fontSize: 16,
        width: '100%',
        textAlign: 'center'
    },
    tableText: {
        fontFamily: 'Mina-Regular',
        width: '100%',
        textAlign: 'center'
    },
    priceContainer:{
        width: '100%',
        height: 60,
        backgroundColor: primaryColor,
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        gap: 12,
        alignItems: 'center',
        borderColor: primaryColor,
        borderRadius: 16,
        bottom: -1,
    },
    manageButtonContainer:{
        padding: 8,
        backgroundColor: "white",
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        borderColor: primaryColor,
        borderRadius: 16,
    },
    manageButtonText:{
        color: primaryColor,
        fontFamily: 'Mina-Bold'
    },
    trashContainer:{
        display: "flex",
        position: "absolute",
        width: 30,
        height: 30,
        right: 5,
        top: 4
    },
    trashIcon:{
        width: 30,
        height: 30,
    },

    /* Modal styles */
    modalContainer: {
        flex: 1,
        justifyContent: 'center',
        alignItems: 'center',
        backgroundColor: 'rgba(0, 0, 0, 0.5)',
    },
    modalContent: {
        width: width * 0.8,
        padding: 20,
        backgroundColor: 'white',
        borderRadius: 20,
        alignItems: 'center',
        borderColor: primaryColor,
        borderWidth: 3,
    },
    modalTitle: {
        fontSize: 24,
        color: primaryColor,
        fontFamily: 'Mina-Bold',
        marginBottom: 20,
    },
    modalInput: {
        width: '100%',
        height: 40,
        borderColor: primaryColor,
        borderWidth: 2,
        borderRadius: 10,
        padding: 10,
        marginBottom: 20,
        fontSize: 16,
        color: primaryColor,
    },
    modalButton: {
        width: '100%',
        padding: 10,
        borderRadius: 10,
        backgroundColor: primaryColor,
        alignItems: 'center',
        marginBottom: 10,
    },
    modalButtonText: {
        color: 'white',
        fontSize: 18,
        fontFamily: 'Mina-Bold',
    },

    /* Sort */
    sortContent: {
        width,
        height: 130,
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
