import { StyleSheet, Dimensions } from "react-native";

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;

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
        width: width,
        paddingBottom: 100,
        
    },
    payForCartButtonBox: {
        width: '100%',
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        
    },
    payForCartButton:{
        backgroundColor: primaryColor,
        flexDirection: "row",
        width: width * 0.7,
        height: 65,
        borderRadius: 10,
        borderColor: '#fff',
        borderWidth: 2,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        position: 'absolute',
        bottom: 20
    },
    payForCartText:{
        color: "white",
        fontSize: 22,
        fontFamily: 'Mina-Bold'
    },
    cashImage:{
        width: 75,
        objectFit: 'contain',
        marginRight: 20,
    },
    cartProduct:{
        display: "flex",
        flexDirection: "row",
        height: 140,
        width: width * 0.95,
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        backgroundColor: "white",
        marginTop: 20
    },
    productImageContainer:{
        backgroundColor: "#d7d7d7",
        width: 140,
        height: 140,
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 20,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        left: -3,
        top: -3,
        zIndex: 1
    },
    productImage:{
        flex: 1,
        width: '100%',
        height: '100%',
        borderRadius: 20,
    },
    productInfoContainer:{
        width: '65%',
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
    },
    nameProductText:{
        color: primaryColor,
        fontSize: 20,
        fontFamily: 'Mina-Bold',
        marginTop: 10,
        overflow: 'hidden'
    },
    sizeContainer:{
        display: 'flex',
        flexDirection: 'row',
    },
    sizeNameText:{
        color: primaryColor,
        fontFamily: 'Mina-Bold',
        fontSize: 16,
    },
    sizeText:{
        color: primaryColor,
        fontFamily: 'Mina-Regular',
        fontSize: 16,
    },
    priceContainer:{
        width: '112%',
        height: 40,
        backgroundColor: primaryColor,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
        padding: 4,
        borderRadius: 17,
        zIndex: -1,
        right: width * 0.05 + 3,
        bottom: 0
    },
    trashContainer:{
        display: "flex",
        position: "absolute",
        width: 35,
        height: 35,
        right: 5,
        bottom: 40
    },
    trashIcon:{
        width: '100%',
        height: '100%',
    },
    quantityContainer: {
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        marginBottom: 4
    },
    quantityButton: {
        backgroundColor: primaryColor,
        width: 24,
        height: 24,
        borderRadius: 13,
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        marginHorizontal: 6,
    },
    quantityText: {
        color: '#fff',
        fontFamily: 'Mina-Bold',
        fontSize: 16,
        textAlignVertical: 'center',
        alignItems: 'center',
    }
});
