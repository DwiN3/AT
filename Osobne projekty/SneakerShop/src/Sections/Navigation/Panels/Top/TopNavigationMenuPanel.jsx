import React from 'react';
import { View, Text, TouchableOpacity, Image } from 'react-native';
import { styles } from '../../Styles/TopNavigationPanel';

export default function TopNavigationPanel({ onPressFilter, onPressSort }) {

    return (
        <View style={styles.body}>
            <View style={styles.mainContainer}>
                <TouchableOpacity onPress={onPressSort}>
                    <Text style={styles.navigateText}>Sortuj</Text>
                </TouchableOpacity>

                <Image style={styles.nameImage} source={require('../../../../../assets/Images/Login/Name.png')} />
                
                <TouchableOpacity onPress={onPressFilter}>
                    <Text style={styles.navigateText}>Filtruj</Text>
                </TouchableOpacity>
            </View>
            <View style={styles.bottomBorder}></View>
        </View>
    );
}
