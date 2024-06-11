import React from 'react';
import { View, TouchableOpacity, Image } from 'react-native';
import { useNavigation } from '@react-navigation/native'; 
import { styles } from '../../Styles/TopNavigationPanel';
import Ionicons from 'react-native-vector-icons/Ionicons';

export default function TopNavigationAccountSettings() {
    const navigation = useNavigation(); 

    return (
        <View style={styles.body}>
            <View style={styles.mainContainer}>                
                <TouchableOpacity onPress={() => navigation.navigate('AccountSettings')} style={styles.box}>
                    <Ionicons name={'arrow-back-outline'} size={40} color= {'#fff'} />
                </TouchableOpacity>
                
                <Image style={styles.nameImage} source={require('../../../../../assets/Images/Login/Name.png')} />
                
                <View style={styles.box}></View>
            </View>
            <View style={styles.bottomBorder}></View>
        </View>
    );
}
