import React from 'react';
import { View, TouchableOpacity, Image } from 'react-native';
import { useNavigation } from '@react-navigation/native'; 
import { styles } from '../../Styles/TopNavigationPanel';

export default function TopNavigationPanel() {
    const navigation = useNavigation(); 

    return (
        <View style={styles.body}>
            <View style={styles.mainContainer}>
                <TouchableOpacity onPress={() => navigation.navigate('MainMenu')}>
                    <Image style={styles.iconsSize} source={require('../../../../../assets/Images/Menu/ArrowIcon.png')} />
                </TouchableOpacity>

                <Image style={styles.nameImage} source={require('../../../../../assets/Images/Login/Name.png')} />
                
                <TouchableOpacity onPress={() => navigation.navigate('ShoppingCart')}>
                    <Image style={styles.iconsSize} source={require('../../../../../assets/Images/Menu/ShopingCartIcon.png')} />
                </TouchableOpacity>
            </View>
            <View style={styles.bottomBorder}></View>
        </View>
    );
}
