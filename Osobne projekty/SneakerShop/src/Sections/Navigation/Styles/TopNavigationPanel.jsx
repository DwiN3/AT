import { StyleSheet } from "react-native";
import {Dimensions} from 'react-native';

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;

export const styles = StyleSheet.create({
    body: {
        backgroundColor: '#c3c3dc',
    },

    mainContainer:{
        width,
        height: 50,
        flexDirection: 'row', 
        justifyContent: 'space-between', 
        alignItems: 'center',
        paddingHorizontal: 15
    },

    bottomBorder:{
        width,
        height: 2,
        backgroundColor: "white",
    },
    
    nameImage:{
        marginTop: 4,
        width: 180,
        height: 30,
    },

    iconsSize:{
        width: 55,
        height: 30,
        marginTop:3,
    },

    navigateText:{
        color: "white",
        width: 55,
        height: 30,
        fontSize: 17,
        marginTop:3,
    },

    box: {
        width: 40,
    }
});
