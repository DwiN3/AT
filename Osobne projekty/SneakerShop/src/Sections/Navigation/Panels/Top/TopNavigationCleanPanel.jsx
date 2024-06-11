import React from 'react';
import { View, Image } from 'react-native';
import { styles } from '../../Styles/TopNavigationPanel';

export default function TopNavigationPanel() {

    return (
        <View style={styles.body}>
            <View style={styles.mainContainer}>
                <View style={{ flex: 1, alignItems: 'center', justifyContent: 'center' }}>
                    <Image style={styles.nameImage} source={require('../../../../../assets/Images/Login/Name.png')} />
                </View>
            </View>
            <View style={styles.bottomBorder}></View>
        </View>
    );
}
