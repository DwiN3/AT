import { StyleSheet } from "react-native";
import {Dimensions} from 'react-native';

const width = Dimensions.get('window').width;
const height = Dimensions.get('window').height;

export const styles = StyleSheet.create({
        container: {
            flex: 1,
            width,
            justifyContent: 'center',
            alignItems: 'center',
            backgroundColor: '#fff',
        },
        image: {
            height: '60%',
            justifyContent: 'center',
            width: width / 1.05,
            resizeMode: 'contain',
        },

        textBox: {
            height: '40%',
            justifyContent: 'center',
            alignItems: 'center',
        },
        title: {
            fontFamily: 'Mina-Bold',
            color: '#e91e62',
            fontSize: 30,
            textAlign: 'center',
            paddingHorizontal: '5%',
            marginBottom: 10,
        },
        description: {
            fontFamily: 'Mina-Regular',
            fontSize: 16,
            color: '#490e75',
            textAlign: 'center',
            paddingHorizontal: '5%',
            marginBottom: 30,
        },

        dot: {
            height: 10,
            borderRadius: 5,
            backgroundColor: '#490e75',
            marginTop: 10,
            marginHorizontal: 8,
        },
        buttonNext: {
            position: "absolute",
            backgroundColor:'#e91e62',
            borderRadius: 100,
            padding: 20,
        },
});