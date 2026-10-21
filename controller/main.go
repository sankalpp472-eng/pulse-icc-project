package main

import (
	"os"

	"k8s.io/apimachinery/pkg/runtime"
	utilruntime "k8s.io/apimachinery/pkg/util/runtime"
	knativeservingv1 "knative.dev/serving/pkg/apis/serving/v1"
	ctrl "sigs.k8s.io/controller-runtime"

	pulsev1alpha1 "pulse-icc-project/controller/api/v1alpha1"
	pulsecontroller "pulse-icc-project/controller/controllers"
)

func main() {
	scheme := runtime.NewScheme()

	utilruntime.Must(pulsev1alpha1.AddToScheme(scheme))
	utilruntime.Must(knativeservingv1.AddToScheme(scheme))

	manager, err := ctrl.NewManager(
		ctrl.GetConfigOrDie(),
		ctrl.Options{
			Scheme: scheme,
		},
	)
	if err != nil {
		ctrl.Log.Error(err, "unable to create manager")
		os.Exit(1)
	}

	reconciler := &pulsecontroller.PrewarmRequestReconciler{
		Client: manager.GetClient(),
	}

	if err := reconciler.SetupWithManager(manager); err != nil {
		ctrl.Log.Error(err, "unable to register PrewarmRequest reconciler")
		os.Exit(1)
	}

	ctrl.Log.Info("Pulse Prewarming Controller starting")

	if err := manager.Start(ctrl.SetupSignalHandler()); err != nil {
		ctrl.Log.Error(err, "problem running manager")
		os.Exit(1)
	}
}
