import { StyleSheet, Dimensions } from "react-native";

const { width, height } = Dimensions.get('window');

const primaryColor = '#411c5d';

export const styles = StyleSheet.create({
    body: {
        backgroundColor: '#c3c3dc',
        alignItems: 'center',
    },
    mainContainer:{
        flexGrow: 1,
        alignItems: 'center',
        width: width,
        paddingBottom: 175,
    },
    productContainer: {
        width: width,
        height: height * 0.4,
        marginVertical: 10,
    },
    carouselImageContainer:{
        backgroundColor: "#d7d7d7",
        width: width,
        height: '100%',
        borderColor: primaryColor,
        borderWidth: 2,
        borderRadius: 20,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },
    carouselImage: {
        flex: 1,
        width: '100%',
        height: '100%',
        borderRadius: 20,
        resizeMode: 'cover',
    },
    categoryText:{
        color: primaryColor,
        fontSize: 20,
        fontFamily: 'Mina-Regular',
    },
    nameProductText:{
        fontFamily: 'Mina-Bold',
        color: primaryColor,
        fontSize: 33,
        marginTop: 15,
        marginHorizontal: 10,
        textAlign: 'center'
    },
    sizeProductContainer:{
        display: 'flex',
        flexDirection: 'row',
        justifyContent: 'center',
        alignItems: 'center',
        height: 50
    },
    sizeProductText:{
        color: primaryColor,
        fontSize: 20,
        fontFamily: 'Mina-Bold',
    },
    detailsHeaderText: {
        color: primaryColor,
        fontSize: 25,
        fontFamily: 'Mina-Bold',
    },
    detailsText: {
        color: primaryColor,
        marginTop: 3,
        fontSize: 15,
        fontFamily: 'Mina-Regular'
    },
    detailsNameText: {
        fontFamily: 'Mina-Bold'
    },
    SelectorContainer:{
        backgroundColor: "#d7d7d7",
        width: width * 0.4,
        height: 35,
        marginLeft: 10,
        borderColor: primaryColor,
        borderWidth: 3,
        borderRadius: 15,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },
    sizePickerContainer:{
        width: '100%',
        height: 10,
    },
    addProductButtonContainer: {
        position: 'absolute',
        bottom: 70,
        width: width * 0.7,
        height: 65,
    },
    addProductButton:{
        backgroundColor: primaryColor,
        flexDirection: "row",
        width: '100%',
        height: '100%',
        borderRadius: 15,
        display: 'flex',
        justifyContent: 'center',
        alignItems: 'center',
    },
    addProductText:{
        color: "white",
        fontSize: 30,
        fontFamily: 'Mina-Bold',
    },
    shopingCartImage:{
        width: 45,
        height: 45,
        marginRight: 20,
    },
    descriptionContainer:{
        width: width * 0.75,
    }
});
