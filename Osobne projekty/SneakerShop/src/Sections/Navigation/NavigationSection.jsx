import React, { useEffect, useState } from 'react';
import { createNativeStackNavigator } from '@react-navigation/native-stack';
import * as Keychain from 'react-native-keychain';

import LoginScreen from '../Login/Screens/LoginScreen';
import SignUpScreen from '../Login/Screens/SignUpScreen';
import ProductScreen from '../User/Screens/ProductScreen';
import MainMenuScreen from '../User/Screens/MainMenuScreen';
import ShoppingCartScreen from '../User/Screens/ShoppingCartScreen';
import OrderApprovalScreen from '../Worker/Screens/OrderApprovalScreen';
import ProductManagementScreen from '../Worker/Screens/ProductManagementScreen';
import AddProductScreen from '../Worker/Screens/AddProductScreen';
import AccountSettingsScreen from '../User/Screens/AccountSettingsScreen';
import ChangePersonalInformationScreen from '../User/Screens/ChangePersonalInformationScreen';
import ChangeDeliveryInformationScreen from '../User/Screens/ChangeDeliveryInformationScreen';
import ChangePasswordScreen from '../User/Screens/ChangePasswordScreen';
import NewsletterSettingsScreen from '../User/Screens/NewsletterSettingsScreen';
import LoadingModal from '../Login/Components/LoadingModal';

const Stack = createNativeStackNavigator();

export default function NavigationSection() {
    const [userRole, setUserRole] = useState(null);
    const [loading, setLoading] = useState(true);

    const getUserRole = async () => {
        try {
            const credentials = await Keychain.getGenericPassword();
            if (credentials) {
                const parsedData = JSON.parse(credentials.password);
                setUserRole(parsedData.roles[0].name);
                // console.log(userRole); /* DO USUNIĘCIA */
                // console.log('Parsed Credentials ns:', parsedData);
            } else {
                console.log('No credentials stored');
            }
        } catch (error) {
            console.error('Failed to load user data', error);
        }  finally {
            setLoading(false);
        }
    };

    useEffect(() => {
        getUserRole()
    });

    if (loading) {
        return (
            <LoadingModal visible={loading}/>
        );
    }

    return (
        <Stack.Navigator initialRouteName={userRole === "User"? "MainMenu": userRole === "Employee"? "AddProduct": "LoginScreen"} screenOptions={{headerShown: false}}>
            <Stack.Screen name="Login" component={LoginScreen} />
            <Stack.Screen name='SignUp' component={SignUpScreen} />
            <Stack.Screen name='Product' component={ProductScreen} />
            <Stack.Screen name='MainMenu' component={MainMenuScreen} />
            <Stack.Screen name='ShoppingCart' component={ShoppingCartScreen} />
            <Stack.Screen name='OrderApproval' component={OrderApprovalScreen} />
            <Stack.Screen name='ProductManagement' component={ProductManagementScreen} />
            <Stack.Screen name='AddProduct' component={AddProductScreen} />
            <Stack.Screen name='AccountSettings' component={AccountSettingsScreen} />
            <Stack.Screen name='ChangePersonalInformation' component={ChangePersonalInformationScreen} />
            <Stack.Screen name='ChangeDeliveryInformatin' component={ChangeDeliveryInformationScreen} />
            <Stack.Screen name='ChangePassword' component={ChangePasswordScreen} />
            <Stack.Screen name='NewsletterSettings' component={NewsletterSettingsScreen} />
        </Stack.Navigator>
    );
}