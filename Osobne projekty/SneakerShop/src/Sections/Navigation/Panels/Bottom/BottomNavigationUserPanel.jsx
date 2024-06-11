import React from 'react';
import { View, TouchableOpacity,Image } from 'react-native';
import { useNavigation } from '@react-navigation/native'; 
import { styles } from '../../Styles/BottomNavigationPanel';
import { Shadow } from 'react-native-shadow-2';
import Ionicons from 'react-native-vector-icons/Ionicons';

export default function BottomNavigationUserPanel() {
    const navigation = useNavigation();

    return (
        <View style={styles.body}>
            <Shadow distance={10} startColor={'#411c5d'}  offset={[1, 1]}>
                <View style={styles.mainContainer}>
                    <TouchableOpacity onPress={() => navigation.navigate('MainMenu')}>
                        <Ionicons name={'home-outline'} size={40} color= {'#411c5d'} style={styles.IconsSize} />
                    </TouchableOpacity>
                    
                    <TouchableOpacity onPress={() => navigation.navigate('ShoppingCart')}>
                        <Ionicons name={'cart-outline'} size={45} color= {'#411c5d'} style={styles.IconsSize} />
                    </TouchableOpacity>

                    <TouchableOpacity onPress={() => navigation.navigate('AccountSettings')}>
                        <Ionicons name={'person-outline'} size={40} color= {'#411c5d'} style={styles.IconsSize} />
                    </TouchableOpacity>
                </View>
            </Shadow>
        </View>
    );
}
