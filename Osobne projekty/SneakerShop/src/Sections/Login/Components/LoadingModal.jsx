import { View, Modal, ActivityIndicator } from 'react-native';
import React, { Component } from 'react';
import { styles } from '../Styles/modalLoadingStyle';

export default function LoadingModal({visible}) {

    return (
        <Modal
        transparent={true}
        animationType={'fade'}
        visible={visible}
        onRequestClose={() => {}}>
        <View style={styles.modalBackground}>
            <ActivityIndicator size="large" color="#C3C7DF" style={{ transform: [{ scale: 2.5 }] }} />
        </View>
    </Modal>
    )
}