import React, { useState } from 'react';
import { View, Text, TouchableOpacity, Image, ScrollView, ToastAndroid, KeyboardAvoidingView } from 'react-native';
import { styles } from '../Styles/ChangeInformationStyle';
import TopNavigationAccountSettings from '../../Navigation/Panels/Top/TopNavigationAccountSettings';
import AccountSettingInput from '../Models/AccountSettingInput';
import Animated, { FadeInDown, FadeInUp } from 'react-native-reanimated';

export default function ChangePasswordScreen() {
    const [password, setPassword] = useState({
        oldPassword: '',
        newPassword: '',
        rePassword: '',
    });


    const [errorMessage, setErrorMessage] = useState ({
        oldPassword: '',
        newPassword: '',
        rePassword: '',
    });
    
    const validate = async () => {
        let errorMessages = {...errorMessage};

        if (password.oldPassword.trim() === '') {
            errorMessages.oldPassword = 'Old password is required!';
        } else if(isOldPasswordCorrect(password.oldPassword)){
            errorMessages.oldPassword = 'Old password is incorrect!';
        } else {
            errorMessages.oldPassword = '';
        }
        if (password.newPassword.trim() === '') {
            errorMessages.newPassword = 'Password is required!';
        } else if (!isStrongPassword(password.newPassword)) {
            errorMessages.newPassword = 'Password must be at least 8 characters long and include one uppercase letter, one lowercase letter, and one number!';
        } else {
            errorMessages.newPassword = '';
        }
        if (password.rePassword.trim() === '') {
            errorMessages.rePassword = 'Re-password is required!';
        } else if (password.newPassword !== password.rePassword) {
            errorMessages.rePassword = 'Passwords must be the same!';
        } else {
            errorMessages.rePassword = '';
        }

        setErrorMessage(errorMessages);
        return errorMessages;
    };

    const isOldPasswordCorrect = (oldPassword) => {
        /* TO DO CHECK PASSWORD IN DATABASE */
        /* return true = correct */
        return false;
    }

    const isStrongPassword = (password) => {
        return /^(?=.*[a-z])(?=.*[A-Z])(?=.*\d).{8,}$/.test(password);
    }

    const handleUpdate = async () => {
        console.log(password);   /* DO USUNIĘCIA */
       
        const errorMessages = await validate();

        if (Object.values(errorMessages).some(value => value !== '')) {
            ToastAndroid.showWithGravity('Wrong Input Data!', ToastAndroid.SHORT, ToastAndroid.CENTER);
            return;
        }

        /* TO DO UPDATE */
        setPassword({
            oldPassword: '',
            newPassword: '',
            rePassword: '',
        });
    }

    const handleChange = (name, value) => {
        setPassword(prevData => ({
            ...prevData,
            [name]: value
        }));
    };

    return (
        <View style={styles.body}>
            <TopNavigationAccountSettings />
            <KeyboardAvoidingView behavior='height' style={styles.form} keyboardVerticalOffset={20}>
                <Animated.ScrollView entering={FadeInUp.duration(500).springify().delay(200)} contentContainerStyle={styles.card}>
                    <View style={styles.header}>
                        <View style={styles.headerTextBox}>
                            <Text style={styles.title}>Change Password</Text>
                        </View>
                        <Image style={styles.stamp} source={require('../../../../assets/Images/User/password.png')}/>
                    </View>
                    <View>
                        <AccountSettingInput text='Old password' value={password.oldPassword} secue={true} handleOnChange={(value) => handleChange('oldPassword', value)} errorMessage={errorMessage.oldPassword} />
                        <AccountSettingInput text='New password' value={password.newPassword} secue={true} handleOnChange={(value) => handleChange('newPassword', value)} errorMessage={errorMessage.newPassword} />
                        <AccountSettingInput text='Re-password' value={password.rePassword} secue={true} handleOnChange={(value) => handleChange('rePassword', value)} errorMessage={errorMessage.rePassword} />
                        <TouchableOpacity style={styles.button} onPress={() => handleUpdate()}>
                            <Text style={styles.title}>Change</Text>
                        </TouchableOpacity>
                    </View>
                </Animated.ScrollView>
            </KeyboardAvoidingView>
        </View>
    );
}