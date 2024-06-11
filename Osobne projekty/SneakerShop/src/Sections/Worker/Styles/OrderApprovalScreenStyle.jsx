import { StyleSheet, Dimensions } from "react-native";

const { width } = Dimensions.get('window');

const primaryColor = '#411c5d';

export const styles = StyleSheet.create({
    body: {
        backgroundColor: '#c3c3dc',
        flex: 1, 
        alignItems: 'center',
    },
    mainContainer:{
        flexGrow: 1,
        alignItems: 'center',
        width: width * 0.95,
    },
    setContainer:{
        backgroundColor: "#d7d7d7",
        width: "100%",
        flex: 1,
        alignItems: 'center',
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        marginVertical: 10,
        padding: 10
    },
    contactContainer: {
        padding: 10
    },
    contactTextLabel: {
        fontFamily: 'Mina-Bold'
    },
    contactText: {
        fontFamily: 'Mina-Regular',
        fontSize: 16,
        color: '#000'
    },
    acceptButton:{
        backgroundColor: primaryColor,
        flexDirection: "row",
        width: width * 0.7,
        height: 65,
        borderRadius: 15,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        marginVertical: 20
    },
    acceptText:{
        color: "white",
        fontSize: 28,
        fontFamily: 'Mina-Bold'
    },
    acceptImage:{
        width: 45,
        height: 45,
        marginRight: 20,
    },
    cartProduct:{
        display: "flex",
        flexDirection: "row",
        height: 130,
        width: '95%',
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        backgroundColor: "white",
        marginBottom: 10
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
        width: '100%',
        height: '100%',
        display: 'flex',
        borderRadius: 20,
    },
    productInfoContainer:{
        width: '65%',
        height: 'auto',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },
    nameProductText:{
        color: primaryColor,
        fontSize: 20,
        textAlign: 'center',
        fontFamily: 'Mina-Bold',
    },
    sizeText:{
        color: primaryColor,
        fontSize: 16,
        fontFamily: 'Mina-Regular'
    },
    quantityText:{
        color: primaryColor,
        fontSize: 16,
        fontFamily: 'Mina-Regular'
    },
    /* Sort */
    sortContent: {
        width,
        height: 85,
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
